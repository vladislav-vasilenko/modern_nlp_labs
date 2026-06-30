from __future__ import annotations

import torch
import torch.nn as nn
from gradients_lab import (
    manual_neuron_backward,
    track_gradient_norms,
    clip_gradients_by_norm_,
)

def main():
    print("=== Gradients Lab: Basic Usage ===")
    
    # 1. Single Neuron Backpropagation
    print("\n--- 1. Manual vs Autograd Single Neuron ---")
    x = torch.tensor([1.0, -2.0, 0.5])
    w = torch.tensor([0.5, 0.2, -1.0])
    b = torch.tensor(0.1)
    y = torch.tensor(1.0)
    
    gx, gw, gb = manual_neuron_backward(x, w, b, y)
    print("Inputs:      x =", x.numpy())
    print("Weights:     w =", w.numpy())
    print("Manual grad_w:", gw.numpy())
    print("Manual grad_b:", gb.item())
    
    # Verify with PyTorch
    x_pt = x.clone().requires_grad_(True)
    w_pt = w.clone().requires_grad_(True)
    b_pt = b.clone().requires_grad_(True)
    
    z = torch.sum(x_pt * w_pt) + b_pt
    a = torch.sigmoid(z)
    loss = 0.5 * (a - y) ** 2
    loss.backward()
    
    print("PyTorch grad_w:", w_pt.grad.numpy())
    print("PyTorch grad_b:", b_pt.grad.item())
    
    # 2. Gradient tracking in sequential model
    print("\n--- 2. Tracking Gradient Norms ---")
    model = nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    
    x_batch = torch.randn(4, 5)
    y_batch = torch.randn(4, 1)
    
    norms = track_gradient_norms(model, x_batch, y_batch)
    for res in norms:
        print(f"Param: {res['name']:<15} Shape: {str(res['shape']):<15} Grad L2 Norm: {res['grad_norm']:.6f}")
        
    # 3. Gradient clipping
    print("\n--- 3. Custom Gradient Clipping ---")
    params = list(model.parameters())
    print("Total norm before clipping:")
    total_norm = torch.sqrt(sum(p.grad.norm(2)**2 for p in params if p.grad is not None)).item()
    print(f"  {total_norm:.6f}")
    
    # Clip to max norm 1.0
    clip_gradients_by_norm_(params, max_norm=1.0)
    
    clipped_norm = torch.sqrt(sum(p.grad.norm(2)**2 for p in params if p.grad is not None)).item()
    print(f"Total norm after clipping (max_norm=1.0):")
    print(f"  {clipped_norm:.6f}")
    print("====================================")


if __name__ == "__main__":
    main()
