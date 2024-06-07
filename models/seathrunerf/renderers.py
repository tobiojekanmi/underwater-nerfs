import torch
from torch import Tensor, nn
from jaxtyping import Float
from typing import Literal, Union, Tuple, Any, Dict
from nerfstudio.cameras.rays import RaySamples


def get_transmittance(delta_density):
    object_transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2)
    object_transmittance = torch.cat(
        [
            torch.zeros(
                (*object_transmittance.shape[:-2], 1, object_transmittance.shape[-1]),
                device=delta_density.device,
            ),
            object_transmittance,
        ],
        dim=-2,
    )
    object_transmittance = torch.exp(-object_transmittance)

    return object_transmittance


class SeathruNeRFRGBRenderer(nn.Module):
    """
    Volumetric RGB rendering of an unnderwater scene.

    Args:
        use_new_rendering_eqs: Whether to use the new rendering equations.
    """

    def combine_rgb(
        self,
        object_rgbs: Float[Tensor, "*bs num_samples 3"],
        medium_rgbs: Float[Tensor, "*bs num_samples 3"],
        direct_coeffs: Float[Tensor, "*bs num_samples 3"],
        backscatter_coeffs: Float[Tensor, "*bs num_samples 3"],
        densities: Float[Tensor, "*bs num_samples 1"],
        ray_samples: RaySamples,
    ) -> Dict[str, Union[Tensor, None]]:
        """
        Render pixel colour along rays using volumetric rendering.

        Args:
            object_rgb: RGB values of object.
            medium_rgb: RGB values of medium.
            medium_bs:  sigma backscatter of medium.
            medium_attn: sigma attenuation of medium.
            densities: Object densities.
            ray_samples: Set of ray samples.
        """

        # starts = ray_samples.frustums.starts.detach()
        deltas = ray_samples.deltas.detach()  # type: ignore

        # Restored Object Color
        delta_densities = deltas * densities
        object_transmittance = get_transmittance(delta_densities)
        object_alphas = 1 - torch.exp(-delta_densities)
        object_weights = object_transmittance * object_alphas
        restored = torch.clamp(torch.sum(object_weights * object_rgbs, dim=-2), 0, 1)

        # Direct Signal
        # direct_attenuation = torch.exp(-direct_coeffs * starts)
        direct_attenuation = get_transmittance(direct_coeffs * deltas)
        direct_weights = object_transmittance * direct_attenuation * object_alphas
        direct_ray = torch.sum(direct_weights * object_rgbs, dim=-2)
        direct = torch.clamp(direct_ray, 0, 1)

        # Backscatter Signal
        # backscattering = torch.exp(-backscatter_coeffs * starts)
        backscattering = get_transmittance(backscatter_coeffs * deltas)
        medium_alphas = 1 - torch.exp(-backscatter_coeffs * deltas)
        backscatter_weights = object_transmittance * backscattering * medium_alphas
        backscatter_ray = torch.sum(backscatter_weights * medium_rgbs, dim=-2)
        backscatter = torch.clamp(backscatter_ray, 0, 1)

        # Degraded Signal
        rgb = torch.clamp(direct_ray + backscatter_ray, 0, 1)

        # Segmentation Mask
        object_mask = torch.clamp(
            torch.sum(object_weights, dim=-2, keepdim=False), 0, 1
        )

        return {
            "rgb": rgb,
            "restored": restored,
            "direct": direct,
            "backscatter": backscatter,
            "medium_rgbs": medium_rgbs,
            "object_transmittance": object_transmittance,
            "object_alphas": object_alphas,
            "object_weights": object_weights,
            "direct_coeffs": direct_coeffs[:, 0, :],
            "backscatter_coeffs": backscatter_coeffs[:, 0, :],
            "object_mask": object_mask,
        }

    def forward(
        self,
        object_rgbs: Float[Tensor, "*bs num_samples 3"],
        medium_rgbs: Float[Tensor, "*bs num_samples 3"],
        direct_coeffs: Float[Tensor, "*bs num_samples 3"],
        backscatter_coeffs: Float[Tensor, "*bs num_samples 3"],
        densities: Float[Tensor, "*bs num_samples 1"],
        ray_samples: RaySamples,
    ) -> Dict[str, Union[Tensor, None]]:

        object_rgbs = torch.nan_to_num(object_rgbs)
        medium_rgbs = torch.nan_to_num(medium_rgbs)
        direct_coeffs = torch.nan_to_num(direct_coeffs)
        backscatter_coeffs = torch.nan_to_num(backscatter_coeffs)

        return self.combine_rgb(
            object_rgbs,
            medium_rgbs,
            direct_coeffs,
            backscatter_coeffs,
            densities,
            ray_samples,
        )
