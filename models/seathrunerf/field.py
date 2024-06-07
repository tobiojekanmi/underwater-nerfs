import torch
from torch import Tensor, nn
from enum import Enum
from typing import Dict, Literal, Optional, Tuple

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import SHEncoding, HashEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.mlp import MLP, MLPWithHashEncoding
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions
from nerfstudio.utils.rich_utils import CONSOLE


class MediumFieldHeadNames(Enum):
    """
    Water Medium MLP outputs for SeaThru-NeRF.
    """

    MEDIUM_RGB = "medium_RGB"
    DIRECT_COEFFS = "direct_coeffs"
    BACKSCATTER_COEFFS = "backscatter_coeffs"


class SeathruNerfField(Field):
    """
    The Modified Seathru-NeRF Field

    Args:
        aabb (Tensor): Parameters of scene aabb bounds.
        num_images (int): Number of images in the dataset.
        num_layers (int, optional): Number of hidden layers.
        hidden_dim (int, optional): Dimension of hidden layers.
        bottleneck_dim (int, optional): Output geo feat dimensions.
        num_levels (int, optional): Number of levels of the hashmap for the base mlp.
        base_res (int, optional): Base resolution of the hashmap for the base mlp.
        max_res (int, optional): Maximum resolution of the hashmap for the base mlp.
        log2_hashmap_size (int, optional): Size of the hashmap for the base mlp.
        num_layers_color (int, optional): Number of hidden layers for color network.
        num_layers_transient (int, optional): Number of hidden layers for transient network.
        features_per_level (int, optional): Number of features per level for the hashgrid.
        hidden_dim_color (int, optional): Dimension of hidden layers for color network.
        hidden_dim_transient (int, optional): Dimension of hidden layers for transient network.
        appearance_embedding_dim (int, optional): Dimension of appearance embedding.
        use_average_appearance_embedding (bool, optional): Whether to use average appearance embedding or zeros for inference.
        spatial_distortion (Optional[SpatialDistortion], optional): Spatial distortion to apply to the scene.
        average_init_density (float, optional): Average initial density.
        implementation (Literal["tcnn", "torch"], optional): Implementation of the field (torch or tcnn).
        object_density_bias (float, optional): Bias to add to object density.
        medium_density_bias (float, optional): Bias to add to medium density.
        use_viewing_direction_for_object_rgb (bool, optional): Whether to use viewing direction for object RGB.
        use_new_model (bool, optional): Whether to use new model.
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
        num_layers: int = 2,
        hidden_dim: int = 64,
        bottleneck_dim: int = 15,
        num_layers_color: int = 2,
        hidden_dim_color: int = 64,
        num_layers_medium: int = 2,
        hidden_dim_medium: int = 64,
        use_appearance_embedding: bool = False,
        appearance_embedding_dim: int = 32,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        average_init_density: float = 1.0,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        object_density_bias: float = 0.0,
        medium_density_bias: float = 0.0,
        use_viewing_direction_for_object_rgb: bool = True,
    ) -> None:
        super().__init__()

        self.register_buffer("aabb", aabb)
        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.step = 0
        self.num_images = num_images
        self.bottleneck_dim = bottleneck_dim
        self.spatial_distortion = spatial_distortion
        self.appearance_embedding_dim = appearance_embedding_dim
        self.use_appearance_embedding = use_appearance_embedding
        if self.use_appearance_embedding:
            if self.appearance_embedding_dim > 0:
                self.embedding_appearance = Embedding(
                    self.num_images, self.appearance_embedding_dim
                )
            else:
                raise ValueError(
                    "Appearance embedding dimension should be greater than 0."
                )
        else:
            self.embedding_appearance = None
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.average_init_density = average_init_density
        self.object_density_bias = object_density_bias
        self.medium_density_bias = medium_density_bias
        self.use_viewing_direction_for_object_rgb = use_viewing_direction_for_object_rgb
        self.medium_colour_activation = nn.Sigmoid()
        self.medium_density_activation = nn.Softplus()

        self.direction_encoding = SHEncoding(
            levels=4,
            implementation=implementation,
        )

        # Object Density MLP
        self.mlp_base = MLPWithHashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=1 + self.bottleneck_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

        # Object Color MLP
        obj_mlp_input_dim = self.bottleneck_dim
        if self.use_viewing_direction_for_object_rgb:
            obj_mlp_input_dim += self.direction_encoding.get_out_dim()
        if self.embedding_appearance is not None:
            obj_mlp_input_dim += self.appearance_embedding_dim
        self.mlp_head = MLP(
            in_dim=obj_mlp_input_dim,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

        # Medium Color, Attenuation and Backscatter MLP
        self.medium_mlp = MLP(
            in_dim=self.direction_encoding.get_out_dim()
            + self.hash_position_encoding.get_out_dim(),
            num_layers=num_layers_medium,
            layer_width=hidden_dim_medium,
            out_dim=9,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """
        Computes and returns the densities.

        Args:
            ray_samples (RaySamples): Ray samples.

        Returns:
            Tuple[Tensor, Tensor]: Tuple containing the computed densities and base MLP output.
        """
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(
                ray_samples.frustums.get_positions(), self.aabb
            )

        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions

        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True

        positions_flat = positions.view(-1, 3)
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(
            h, [1, self.bottleneck_dim], dim=-1
        )
        density_before_activation = density_before_activation + self.object_density_bias
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters. Adadpted from Nerfacto.
        density = self.average_init_density * trunc_exp(
            density_before_activation.to(positions)
        )
        density = density * selector[..., None]

        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        """
        Computes and returns the outputs.

        Args:
            ray_samples (RaySamples): Ray samples.
            density_embedding (Optional[Tensor], optional): Density embedding. Defaults to None.

        Returns:
            Dict[FieldHeadNames, Tensor]: Dictionary containing the computed outputs.
        """
        assert density_embedding is not None
        outputs = {}

        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()

        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        directions_encoded = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # Object Appearance
        embedded_appearance = None
        if self.embedding_appearance is not None:
            if self.training:
                embedded_appearance = self.embedding_appearance(camera_indices)
            else:
                if self.use_average_appearance_embedding:
                    embedded_appearance = torch.ones(
                        (*directions.shape[:-1], self.appearance_embedding_dim),
                        device=directions.device,
                    ) * self.embedding_appearance.mean(dim=0)
                else:
                    embedded_appearance = torch.zeros(
                        (*directions.shape[:-1], self.appearance_embedding_dim),
                        device=directions.device,
                    )

        # Object Color Forward Pass
        obj_mlp_input = density_embedding.view(-1, self.bottleneck_dim)
        if self.use_viewing_direction_for_object_rgb:
            obj_mlp_input = torch.cat([obj_mlp_input, directions_encoded], dim=-1)
        if embedded_appearance is not None:
            obj_mlp_input = torch.cat(
                [
                    obj_mlp_input,
                    embedded_appearance.view(-1, self.appearance_embedding_dim),
                ],
                dim=-1,
            )

        rgb_object = (
            self.mlp_head(obj_mlp_input).view(*outputs_shape, -1).to(directions)
        )
        outputs.update({FieldHeadNames.RGB: rgb_object})

        # Medium MLP forward pass
        medium_mlp_input = torch.cat(
            [
                self.ray_origins,
                directions_encoded,
            ],
            dim=-1,
        ).view(*outputs_shape, -1)[:, 0]

        medium_out = self.medium_mlp(medium_mlp_input)

        medium_out = self.medium_mlp(directions_encoded.view(*outputs_shape, -1)[:, 0])

        medium_rgb = (
            self.medium_colour_activation(medium_out[..., :3])
            .view(*outputs_shape, -1)
            .to(directions)
        )
        backscatter_coeffs = (
            self.medium_density_activation(
                medium_out[..., 3:6] + self.medium_density_bias
            )
            .view(*outputs_shape, -1)
            .to(directions)
        )
        direct_coeffs = (
            self.medium_density_activation(
                medium_out[..., 6:] + self.medium_density_bias
            )
            .view(*outputs_shape, -1)
            .to(directions)
        )

        outputs.update({MediumFieldHeadNames.MEDIUM_RGB: medium_rgb})
        outputs.update({MediumFieldHeadNames.DIRECT_COEFFS: backscatter_coeffs})
        outputs.update({MediumFieldHeadNames.BACKSCATTER_COEFFS: direct_coeffs})

        return outputs
