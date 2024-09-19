import torch
from enum import Enum

class SparsificationMethod(str, Enum):
    magnitude = "magnitude"
    random = "random"
    rescaled_random = "rescaled_random"
    differentiable_mask = "differentiable_mask"

def differentiable_mask_l1_penalty(tensor: torch.Tensor, mask: torch.nn.Parameter, l1_lambda: float) -> torch.Tensor:
    """
    Apply a differentiable mask with L1 penalty to the tensor.

    Args:
        tensor (torch.Tensor): The original weight tensor.
        mask (torch.nn.Parameter): The mask tensor (trainable).
        l1_lambda (float): Regularization strength for the L1 penalty.

    Returns:
        torch.Tensor: The masked tensor.
        torch.Tensor: The L1 penalty (to be added to the loss function).
    """
    mask = torch.sigmoid(mask)

    masked_tensor = tensor * mask

    l1_penalty = l1_lambda * torch.norm(mask, p=1)

    return masked_tensor, l1_penalty

def sparsify(
    tensor: torch.Tensor, density: float, method: SparsificationMethod, mask: torch.nn.Parameter = None, l1_lambda: float = 0.01
) -> torch.Tensor:
    if method == SparsificationMethod.magnitude:
        return magnitude(tensor, density=density)
    elif method == SparsificationMethod.random:
        return bernoulli(tensor, density=density, rescale=False)
    elif method == SparsificationMethod.rescaled_random:
        return bernoulli(tensor, density=density, rescale=True)
    elif method == SparsificationMethod.differentiable_mask:
        assert mask is not None, "Mask must be provided for differentiable mask method"
        return differentiable_mask_l1_penalty(tensor, mask, l1_lambda)
    else:
        raise NotImplementedError(method)
