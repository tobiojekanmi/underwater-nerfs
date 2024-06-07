
from typing import Float
import torch
import os
from torch import Tensor


def get_transmittance(
    deltas: Tensor, 
    densities: Float[Tensor, "*bs num_samples 1"],
    ) -> Float[Tensor, "*bs num_samples 1"]:
    """
    Computes and returns the transmittance for each ray sample.

    Args:
        deltas: Distance between each ray sample.
        densities: Densities of each ray sample.
    """
    delta_density = deltas * densities
    transmittance_object = torch.cumsum(delta_density[..., :-1, :], dim=-2)
    transmittance_object = torch.cat(
        [
            torch.zeros(
                (*transmittance_object.shape[:-2], 1, transmittance_object.shape[-1]),
                device=transmittance_object.device,
            ),
            transmittance_object,
        ],
        dim=-2,
    )
    transmittance_object = torch.exp(-transmittance_object)

    return transmittance_object


def add_water(
    img: Tensor, depth: Tensor, beta_D: Tensor, beta_B: Tensor, B_inf: Tensor
) -> Tensor:
    """
    Add water effect to image.
    Image formation model from https://openaccess.thecvf.com/content_cvpr_2018/papers/Akkaynak_A_Revised_Underwater_CVPR_2018_paper.pdf (Eq. 20).

    Args:
        img: image to add water effect to.
        beta_D: depth map.
        beta_B: background map.
        B_inf: background image.

    Returns:
        Image with water effect.
    """  # noqa: E501

    depth = depth.repeat_interleave(3, dim=-1)
    I_out = img * torch.exp(-beta_D * depth) + B_inf * (1 - torch.exp(-beta_B * depth))

    return I_out
