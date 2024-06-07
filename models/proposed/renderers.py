import torch
from torch import Tensor, nn
from jaxtyping import Float
from typing import Dict
from nerfstudio.cameras.rays import RaySamples


def get_transmittance(delta_density):
    transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2)
    transmittance = torch.cat(
        [
            torch.zeros(
                (*transmittance.shape[:-2], 1, transmittance.shape[-1]),
                device=delta_density.device,
            ),
            transmittance,
        ],
        dim=-2,
    )
    transmittance = torch.exp(-transmittance)

    return torch.nan_to_num(transmittance)


class UWNeRFRGBRenderer(nn.Module):
    """
    Volumetric RGB rendering of an unnderwater scene.
    """

    def combine_rgb(
        self,
        densities: Float[Tensor, "*bs num_samples 1"],  # noqa: F722
        rgbs: Float[Tensor, "*bs num_samples 3"],  # noqa: F722
        veiling_light: Float[Tensor, "*bs num_samples 3"],  # noqa: F722
        direct_coeffs: Float[Tensor, "*bs num_samples 3"],  # noqa: F722
        backscatter_coeffs: Float[Tensor, "*bs num_samples 3"],  # noqa: F722
        ray_samples: RaySamples,
    ) -> Dict[str, Tensor]:
        """
        Render pixel colour along rays using volumetric rendering.

        Args:
            densities: volumetric densities or scattering coefficients.
            rgbs: Volumetric RGBs for the Scene.
            veiling_light: Medium RGBs.
            direct_coeffs: Direct attenuation coefficients.
            backscatter_coeffs: Backscattering coefficients.
            ray_samples: Ray samples.
        """
        deltas = ray_samples.deltas.detach()  # type: ignore
        steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) * 0.5

        # Actual Object Color Composition
        object_delta_density = deltas * densities
        object_transmittance = get_transmittance(object_delta_density)
        object_alphas = 1 - torch.exp(-object_delta_density)
        object_weights = object_alphas * object_transmittance
        restored = torch.clamp(torch.sum(object_weights * rgbs, dim=-2), 0, 1)

        # Direct Signal Color Composition
        direct_attenuation = get_transmittance(deltas * direct_coeffs)
        direct_ray = torch.sum(object_weights * direct_attenuation * rgbs, dim=-2)
        direct = torch.clamp(direct_ray, 0, 1)

        # Object Backscatter Color Composition
        backscatter_attenuation = 1 - get_transmittance(deltas * backscatter_coeffs)
        backscatter_weights = object_weights * backscatter_attenuation
        object_backscatter_ray = torch.sum(backscatter_weights * veiling_light, dim=-2)
        object_backscatter = torch.clamp(object_backscatter_ray, 0, 1)

        # Water Column Backscatter Color Composition (No Object)
        object_mask = torch.clamp(
            torch.sum(object_weights, dim=-2, keepdim=False), 0, 1
        )
        backscatter_residual = 1 - object_mask
        medium_backscatter_ray = backscatter_residual * veiling_light[:, 0, :]
        medium_backscatter = torch.clamp(medium_backscatter_ray, 0, 1)

        # Final Backscatter Color Composition
        backscatter_ray = object_backscatter_ray + medium_backscatter_ray
        backscatter = torch.clamp(backscatter_ray, 0, 1)

        # Final Underwater RGB
        rgb = torch.clamp(direct_ray + backscatter_ray, 0, 1)

        return {
            "rgb": rgb,
            "restored": restored,
            "direct": direct,
            "veiling_light": veiling_light[:, 0, :],
            "backscatter": backscatter,
            "object_backscatter": object_backscatter,
            "medium_backscatter": medium_backscatter,
            "object_transmittance": object_transmittance,
            "object_alphas": object_alphas,
            "object_weights": object_weights,
            "direct_coeffs": direct_coeffs[:, 0, :],
            "backscatter_coeffs": backscatter_coeffs[:, 0, :],
            "steps": steps,
            "object_mask": object_mask,
        }

    def forward(
        self,
        densities: Float[Tensor, "*bs num_samples 1"],  # noqa: F722
        rgbs: Float[Tensor, "*bs num_samples 3"],  # noqa: F722
        veiling_light: Float[Tensor, "*bs num_samples 3"],  # noqa: F722
        direct_coeffs: Float[Tensor, "*bs num_samples 3"],  # noqa: F722
        backscatter_coeffs: Float[Tensor, "*bs num_samples 3"],  # noqa: F722
        ray_samples: RaySamples,
    ) -> Dict[str, Tensor]:

        rgbs = torch.nan_to_num(rgbs)
        veiling_light = torch.nan_to_num(veiling_light)
        direct_coeffs = torch.nan_to_num(direct_coeffs)
        backscatter_coeffs = torch.nan_to_num(backscatter_coeffs)

        return self.combine_rgb(
            densities,
            rgbs,
            veiling_light,
            direct_coeffs,
            backscatter_coeffs,
            ray_samples,
        )
