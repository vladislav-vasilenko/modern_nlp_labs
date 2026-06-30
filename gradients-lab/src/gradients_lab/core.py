from __future__ import annotations

import torch
import torch.nn as nn


def manual_neuron_backward(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the forward and manual backward pass for a single neuron:
    z = sum(x * w) + b
    a = sigmoid(z)
    loss = 0.5 * (a - y)^2
    
    Args:
        x: Input tensor of shape (D,) or scalar
        w: Weights tensor of shape (D,) or scalar
        b: Bias tensor (scalar)
        y: Target label (scalar)
        
    Returns:
        (grad_x, grad_w, grad_b) tensors matching original inputs
    """
    # Ensure inputs are floats for gradients
    x = x.detach().clone().float()
    w = w.detach().clone().float()
    b = b.detach().clone().float()
    y = y.detach().clone().float()
    
    # Forward pass
    z = torch.sum(x * w) + b
    a = torch.sigmoid(z)
    
    # Derivatives
    d_loss_d_a = a - y
    d_a_d_z = a * (1.0 - a)
    d_loss_d_z = d_loss_d_a * d_a_d_z
    
    # Gradients
    grad_x = d_loss_d_z * w
    grad_w = d_loss_d_z * x
    grad_b = d_loss_d_z
    
    return grad_x, grad_w, grad_b


def track_gradient_norms(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target: torch.Tensor,
    loss_fn: nn.Module | None = None
) -> list[dict[str, any]]:
    """
    Performs a forward and backward pass, then tracks the L2 norm of the gradients
    for each parameter in the model.
    
    Args:
        model: PyTorch model (e.g. nn.Sequential)
        input_tensor: Model input tensor
        target: Target tensor
        loss_fn: Optional loss function, defaults to nn.MSELoss()
        
    Returns:
        A list of dictionaries with keys:
        - "name": parameter name
        - "shape": parameter shape tuple
        - "grad_norm": float value of gradient L2 norm (or 0.0 if grad is None)
        - "param_norm": float value of parameter weights L2 norm
    """
    # Zero out existing gradients
    model.zero_grad()
    
    # Forward pass
    output = model(input_tensor)
    
    # Loss
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    loss = loss_fn(output, target)
    
    # Backward pass
    loss.backward()
    
    # Collect gradient norms
    results = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm(2).item()
        else:
            grad_norm = 0.0
        
        param_norm = param.norm(2).item()
        
        results.append({
            "name": name,
            "shape": tuple(param.shape),
            "grad_norm": grad_norm,
            "param_norm": param_norm,
        })
    return results


def clip_gradients_by_value_(parameters, clip_value: float) -> None:
    """
    Clips parameter gradients in-place to the range [-clip_value, clip_value].
    Equivalent to torch.nn.utils.clip_grad_value_.
    
    Args:
        parameters: An iterable of Tensors or a single Tensor
        clip_value: The maximum allowed value for any individual gradient element
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    clip_value = float(clip_value)
    for p in parameters:
        if p.grad is not None:
            p.grad.clamp_(min=-clip_value, max=clip_value)


def clip_gradients_by_norm_(parameters, max_norm: float, norm_type: float = 2.0) -> float:
    """
    Clips gradient norm of an iterable of parameters.
    The norm is computed over all gradients together, as if they were concatenated into a single vector.
    Gradients are modified in-place.
    
    Args:
        parameters: An iterable of Tensors or a single Tensor
        max_norm: Maximum norm of the gradients
        norm_type: Type of the used p-norm. Can be 'inf' for infinity norm or any float.
        
    Returns:
        Total norm of the parameters (viewed as a single vector) before clipping.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    # Filter parameters with gradients
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return 0.0
    
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    
    if norm_type == float('inf'):
        total_norm = max(g.data.abs().max().item() for g in grads)
    else:
        total_norm = 0.0
        for g in grads:
            param_norm = g.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1.0 / norm_type)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for g in grads:
            g.data.mul_(clip_coef)
            
    return total_norm
