from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformSampler,
)
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from .field import UWNerfField, MediumFieldHeadNames
from .renderers import UWNeRFRGBRenderer
from .losses import transmittance_loss


@dataclass
class UWNerfModelConfig(ModelConfig):
    """SeathruNerf Model Config"""

    _target: Type = field(default_factory=lambda: UWNerfModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 10.0
    """How far along the ray to stop sampling."""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    base_res: int = 16
    """Resolution of the base grid for the hashgrid."""
    max_res: int = 8192
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    features_per_level: int = 2
    """How many hashgrid features per level"""
    num_layers: int = 2
    """Number of layers for the density network"""
    hidden_dim: int = 256
    """Dimension of hidden layers for the density network"""
    bottleneck_dim: int = 63
    """Dimension of the feature vector output from the density network"""
    num_layers_color: int = 2
    """Number of layers for the color network"""
    hidden_dim_color: int = 256
    """Dimension of hidden layers for color network"""
    num_layers_medium: int = 2
    """Number of layers for the medium network"""
    hidden_dim_medium: int = 128
    """Dimension of hidden layers for medium network"""
    average_init_density: float = 1.0
    """Average initial density output from MLP. """
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 128)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 64
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 1000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {
                "hidden_dim": 16,
                "log2_hashmap_size": 17,
                "num_levels": 5,
                "max_res": 128,
                "use_linear": False,
            },
            {
                "hidden_dim": 16,
                "log2_hashmap_size": 17,
                "num_levels": 5,
                "max_res": 256,
                "use_linear": False,
            },
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 5000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Which implementation to use for the model."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    use_distortion_loss: bool = True
    """Whether to use distortion loss."""
    distortion_loss_mult: float = 0.0001
    """Distortion loss multiplier."""
    use_transmittance_loss: bool = True
    """Whether to use transmittance loss to model the scene as a complete opaque object."""
    transmittance_loss_mult: float = 0.0001
    """Object transmittance loss multiplier."""
    transmittance_loss_beta_prior: float = 10.0


class UWNerfModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: UWNerfModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Fields
        self.field = UWNerfField(
            self.scene_box.aabb,
            num_images=self.num_train_data,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            num_layers=self.config.num_layers,
            hidden_dim=self.config.hidden_dim,
            bottleneck_dim=self.config.bottleneck_dim,
            num_layers_color=self.config.num_layers_color,
            hidden_dim_color=self.config.hidden_dim_color,
            num_layers_medium=self.config.num_layers_medium,
            hidden_dim_medium=self.config.hidden_dim_medium,
            spatial_distortion=scene_contraction,
            average_init_density=self.config.average_init_density,
            implementation=self.config.implementation,
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations

        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert (
                len(self.config.proposal_net_args_list) == 1
            ), "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                average_init_density=self.config.average_init_density,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[
                    min(i, len(self.config.proposal_net_args_list) - 1)
                ]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    average_init_density=self.config.average_init_density,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend(
                [network.density_fn for network in self.proposal_networks]
            )

        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(
                    step,
                    [0, self.config.proposal_warmup],
                    [0, self.config.proposal_update_every],
                ),
                1,
                self.config.proposal_update_every,
            )

        # Change proposal network initial sampler if uniform
        initial_sampler = (
            None  # None is for piecewise as default (see ProposalNetworkSampler)
        )
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(
                single_jitter=self.config.use_single_jitter
            )

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # renderers
        self.renderer_rgb = UWNeRFRGBRenderer()
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="median")
        self.renderer_expected_depth = DepthRenderer(method="expected")

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                self.step = step
                train_frac = np.clip(step / N, 0, 1)
                self.step = step

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
            ray_bundle, density_fns=self.density_fns
        )

        field_outputs = self.field.forward(ray_samples)

        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(
                field_outputs, ray_samples
            )

        outs = self.renderer_rgb(
            densities=field_outputs[FieldHeadNames.DENSITY],
            rgbs=field_outputs[FieldHeadNames.RGB],
            veiling_light=field_outputs[MediumFieldHeadNames.VEILING_LIGHT],  # type: ignore
            direct_coeffs=field_outputs[
                MediumFieldHeadNames.DIRECT_COEFFS
            ],  # type: ignore
            backscatter_coeffs=field_outputs[
                MediumFieldHeadNames.BACKSCATTER_COEFFS
            ],  # type: ignore
            ray_samples=ray_samples,
        )

        weights = outs["object_weights"]
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(
            weights=weights, ray_samples=ray_samples
        )
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
            "densities": field_outputs[FieldHeadNames.DENSITY],
        }
        outputs.update(outs)

        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(
                weights=weights_list[i], ray_samples=ray_samples_list[i]
            )

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        if self.training and self.config.use_distortion_loss:
            metrics_dict["distortion"] = distortion_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}

        gt_rgb = batch["image"].to(self.device)
        pred_rgb = outputs["rgb"]
        loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb)

        if self.training:
            loss_dict["interlevel_loss"] = (
                self.config.interlevel_loss_mult
                * interlevel_loss(outputs["weights_list"], outputs["ray_samples_list"])
            )

            if self.config.use_transmittance_loss:
                loss_dict["transmittance_loss"] = (
                    self.config.transmittance_loss_mult
                    * transmittance_loss(
                        transmittance=outputs["object_weights"],
                        beta=self.config.transmittance_loss_beta_prior,
                    )
                )

            if self.config.use_distortion_loss:
                assert metrics_dict is not None and "distortion" in metrics_dict
                loss_dict["distortion_loss"] = (
                    self.config.distortion_loss_mult * metrics_dict["distortion"]
                )

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )
        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
        }
        if outputs["expected_depth"] is not None:
            images_dict["expected_depth"] = colormaps.apply_depth_colormap(
                outputs["expected_depth"],
                accumulation=outputs["accumulation"],
            )
        if outputs["restored"] is not None and outputs["direct"] is not None:
            images_dict["direct_restored"] = torch.cat(
                [gt_rgb, outputs["direct"], outputs["restored"]], dim=1
            )

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
