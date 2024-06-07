import torch
from torch import Tensor
from jaxtyping import Float


def transmittance_loss(
    transmittance: Float[Tensor, "*bs num_samples 1"], beta: float
) -> torch.Tensor:
    """
    Computes and returns the accumulation_loss.

    Args:
        transmittance: Transmittances of object.
        factor: factor to control the weight of the two distributions
    """
    P = torch.exp(-torch.abs(transmittance) / 0.1) + beta * torch.exp(
        -torch.abs(1 - transmittance) / 0.1
    )
    loss = -torch.log(P)
    return loss.abs().mean()
