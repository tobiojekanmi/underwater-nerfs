import torch
from torch import Tensor, nn
from enum import Enum
from typing import Dict, Literal, Optional, Tuple

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.encodings import HashEncoding, SHEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.mlp import MLP  # , MLPWithHashEncoding
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions
import torch.nn.functional as F


class MediumFieldHeadNames(Enum):
    """
    Additional field head names for the water medium field.
    """

    DIRECT_COEFFS = "direct_attenuation_coefficients"
    BACKSCATTER_COEFFS = "backscatter_attenuation_coefficients"
    VEILING_LIGHT = "veiling_light"


class UWNerfField(Field):
    """
    Underwater NeRF Field

    Args:
        aabb (Tensor): Parameters of scene aabb bounds.
        num_images (int): Number of images in the dataset.
        num_levels (int, optional): Number of levels of the hashmap for the base mlp.
        base_res (int, optional): Base resolution of the hashmap for the base mlp.
        max_res (int, optional): Maximum resolution of the hashmap for the base mlp.
        log2_hashmap_size (int, optional): Size of the hashmap for the base mlp.
        features_per_level (int, optional): Number of features per level for the hashgrid.
        num_layers (int, optional): Number of hidden layers.
        hidden_dim (int, optional): Dimension of hidden layers.
        bottleneck_dim (int, optional): Output geo feat dimensions.
        num_layers_color (int, optional): Number of hidden layers for color network.
        hidden_dim_color (int, optional): Dimension of hidden layers for color network.
        num_layers_medium (int, optional): Number of hidden layers for medium parameters.
        hidden_dim_medium (int, optional): Dimension of hidden layers for medium parameters.
        spatial_distortion (Optional[SpatialDistortion], optional): Spatial distortion. Defaults to None.
        average_init_density (float, optional): Average initial density. Defaults to 1.0.
        implementation (Literal["tcnn", "torch"], optional): Implementation type. Defaults to "tcnn".
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
        num_layers_color: int = 3,
        hidden_dim_color: int = 128,
        num_layers_medium: int = 2,
        hidden_dim_medium: int = 32,
        spatial_distortion: Optional[SpatialDistortion] = None,
        average_init_density: float = 1.0,
        implementation: Literal["tcnn", "torch"] = "tcnn",
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
        self.average_init_density = average_init_density

        # Encoders
        self.hash_position_encoding = HashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation=implementation,
        )
        self.she_direction_encoding = SHEncoding(
            levels=4,
            implementation=implementation,
        )

        # Object Density MLP
        self.mlp_base = MLP(
            in_dim=self.hash_position_encoding.get_out_dim(),
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=1 + self.bottleneck_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

        # Object Color MLP
        self.mlp_head = MLP(
            in_dim=self.bottleneck_dim + self.she_direction_encoding.get_out_dim(),
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

        # Medium Parameters
        medium_mlp_input_dim = self.she_direction_encoding.get_out_dim()
        medium_mlp_input_dim += self.hash_position_encoding.get_out_dim()
        self.mlp_veiling_light = MLP(
            in_dim=medium_mlp_input_dim,
            num_layers=num_layers_medium,
            layer_width=hidden_dim_medium,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )
        self.mlp_attenuation_coefficients = MLP(
            in_dim=medium_mlp_input_dim,
            num_layers=num_layers_medium,
            layer_width=hidden_dim_medium * 2,
            out_dim=6,
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

            ray_origins = self.spatial_distortion(ray_samples.frustums.origins)
            ray_origins = (ray_origins + 2.0) / 4.0

        else:
            positions = SceneBox.get_normalized_positions(
                ray_samples.frustums.get_positions(), self.aabb
            )
            ray_origins = SceneBox.get_normalized_positions(
                ray_samples.frustums.origins, self.aabb
            )

        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True

        positions_flat = self.hash_position_encoding(positions.view(-1, 3))
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(
            h, [1, self.bottleneck_dim], dim=-1
        )
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = self.average_init_density * trunc_exp(
            density_before_activation.to(positions)
        )
        density = density * selector[..., None]

        # Encode the ray origins
        self.ray_origins = self.hash_position_encoding(ray_origins.view(-1, 3))

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

        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        directions_encoded = self.she_direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # Object Color MLP Forward Pass
        obj_color_mlp_input = torch.cat(
            [density_embedding.view(-1, self.bottleneck_dim), directions_encoded],
            dim=-1,
        )
        rgb = self.mlp_head(obj_color_mlp_input).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB: rgb})

        # Medium Parameters
        medium_mlp_input = torch.cat(
            [
                self.ray_origins,
                directions_encoded,
            ],
            dim=-1,
        ).view(*outputs_shape, -1)[:, 0]

        veiling_light = (
            F.sigmoid(self.mlp_veiling_light(medium_mlp_input))
            .unsqueeze(-2)
            .expand_as(rgb)
            .to(directions)
        )
        attenuation_coefficients = self.mlp_attenuation_coefficients(medium_mlp_input)
        direct_coefficients = (
            F.softplus(attenuation_coefficients[..., :3])
            .view(*outputs_shape[:-1], -1)
            .unsqueeze(-2)
            .expand_as(rgb)
            .to(directions)
        )
        backscatter_coefficients = (
            F.softplus(attenuation_coefficients[..., 3:6])
            .view(*outputs_shape[:-1], -1)
            .unsqueeze(-2)
            .expand_as(rgb)
            .to(directions)
        )

        outputs.update({MediumFieldHeadNames.VEILING_LIGHT: veiling_light})
        outputs.update({MediumFieldHeadNames.DIRECT_COEFFS: direct_coefficients})
        outputs.update(
            {MediumFieldHeadNames.BACKSCATTER_COEFFS: backscatter_coefficients}
        )

        return outputs
