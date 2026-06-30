from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from gradients_lab import (
    manual_neuron_backward,
    track_gradient_norms,
    clip_gradients_by_value_,
    clip_gradients_by_norm_,
)


def test_manual_neuron_backward():
    # Test case 1: Scalar inputs
    x = torch.tensor(1.5, requires_grad=False)
    w = torch.tensor(-2.0, requires_grad=False)
    b = torch.tensor(0.5, requires_grad=False)
    y = torch.tensor(1.0, requires_grad=False)
    
    # PyTorch Autograd reference
    x_pt = x.clone().requires_grad_(True)
    w_pt = w.clone().requires_grad_(True)
    b_pt = b.clone().requires_grad_(True)
    y_pt = y.clone()
    
    z = x_pt * w_pt + b_pt
    a = torch.sigmoid(z)
    loss = 0.5 * (a - y_pt) ** 2
    loss.backward()
    
    # Manual
    gx, gw, gb = manual_neuron_backward(x, w, b, y)
    
    assert torch.allclose(gx, x_pt.grad, atol=1e-6)
    assert torch.allclose(gw, w_pt.grad, atol=1e-6)
    assert torch.allclose(gb, b_pt.grad, atol=1e-6)
    
    # Test case 2: Vector inputs
    x_vec = torch.tensor([0.5, -1.0, 2.0])
    w_vec = torch.tensor([0.2, 0.4, -0.1])
    b_scalar = torch.tensor(-0.3)
    y_scalar = torch.tensor(0.0)
    
    x_pt_vec = x_vec.clone().requires_grad_(True)
    w_pt_vec = w_vec.clone().requires_grad_(True)
    b_pt_scalar = b_scalar.clone().requires_grad_(True)
    
    z_vec = torch.sum(x_pt_vec * w_pt_vec) + b_pt_scalar
    a_vec = torch.sigmoid(z_vec)
    loss_vec = 0.5 * (a_vec - y_scalar) ** 2
    loss_vec.backward()
    
    gx_vec, gw_vec, gb_scalar = manual_neuron_backward(x_vec, w_vec, b_scalar, y_scalar)
    
    assert torch.allclose(gx_vec, x_pt_vec.grad, atol=1e-6)
    assert torch.allclose(gw_vec, w_pt_vec.grad, atol=1e-6)
    assert torch.allclose(gb_scalar, b_pt_scalar.grad, atol=1e-6)


def test_clip_gradients_by_value():
    p1 = torch.tensor([1.2, -0.8, 0.1]).requires_grad_(True)
    p2 = torch.tensor([-2.5, 3.0]).requires_grad_(True)
    
    # Mock gradients
    p1.grad = torch.tensor([2.0, -1.5, 0.2])
    p2.grad = torch.tensor([-0.8, 3.5])
    
    # PyTorch reference for verification
    p1_ref = p1.clone()
    p2_ref = p2.clone()
    p1_ref.grad = p1.grad.clone()
    p2_ref.grad = p2.grad.clone()
    
    nn.utils.clip_grad_value_([p1_ref, p2_ref], clip_value=1.0)
    
    # Custom clip
    clip_gradients_by_value_([p1, p2], clip_value=1.0)
    
    assert torch.allclose(p1.grad, p1_ref.grad)
    assert torch.allclose(p2.grad, p2_ref.grad)
    
    # Check bounds
    assert torch.all(p1.grad <= 1.0) and torch.all(p1.grad >= -1.0)
    assert torch.all(p2.grad <= 1.0) and torch.all(p2.grad >= -1.0)


def test_clip_gradients_by_norm():
    p1 = torch.tensor([1.0, 2.0]).requires_grad_(True)
    p2 = torch.tensor([-1.0, 0.5]).requires_grad_(True)
    
    # Mock gradients
    p1.grad = torch.tensor([3.0, 4.0]) # Norm is 5.0
    p2.grad = torch.tensor([0.0, 0.0])
    
    # Combined gradient vector is [3.0, 4.0, 0.0, 0.0] -> Norm is 5.0
    
    p1_ref = p1.clone()
    p2_ref = p2.clone()
    p1_ref.grad = p1.grad.clone()
    p2_ref.grad = p2.grad.clone()
    
    ref_norm = nn.utils.clip_grad_norm_([p1_ref, p2_ref], max_norm=2.0)
    
    # Custom clip
    custom_norm = clip_gradients_by_norm_([p1, p2], max_norm=2.0)
    
    assert abs(ref_norm - custom_norm) < 1e-5
    assert torch.allclose(p1.grad, p1_ref.grad, atol=1e-5)
    assert torch.allclose(p2.grad, p2_ref.grad, atol=1e-5)
    
    # Scaled check: norm should be capped at 2.0
    # Combined: p1.grad should be [3.0, 4.0] * (2/5) = [1.2, 1.6]
    assert torch.allclose(p1.grad, torch.tensor([1.2, 1.6]))


def test_track_gradient_norms():
    # Define a simple 3-layer model
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1)
    )
    
    input_tensor = torch.randn(5, 2)
    target = torch.randn(5, 1)
    
    results = track_gradient_norms(model, input_tensor, target)
    
    assert len(results) == 4 # weights + bias for 2 linear layers
    for res in results:
        assert "name" in res
        assert "shape" in res
        assert "grad_norm" in res
        assert "param_norm" in res
        assert isinstance(res["grad_norm"], float)
        assert isinstance(res["param_norm"], float)
        assert res["grad_norm"] >= 0.0
