import torch
from torch import Tensor
from jaxtyping import Float


def accumulation_loss(
    transmittance_object: Float[Tensor, "*bs num_samples 1"], beta: float
) -> torch.Tensor:
    """
    Computes and returns the accumulation_loss.

    Args:
        transmittance_object: Transmittances of object.
        factor: factor to control the weight of the two distributions
    """
    P = torch.exp(-torch.abs(transmittance_object) / 0.1) + beta * torch.exp(
        -torch.abs(1 - transmittance_object) / 0.1
    )
    loss = -torch.log(P)
    return loss.mean()


def reconstruction_loss(gt_rgb: torch.Tensor, pred_rgb: torch.Tensor) -> torch.Tensor:
    """
    Computes and returns the reconstruction loss.

    Args:
        gt_rgb: Ground truth RGB.
        pred_rgb: Predicted RGB.
    """
    inner = torch.square((pred_rgb - gt_rgb) / (pred_rgb.detach() + 1e-3))
    return inner.mean() # torch.mean(inner)
