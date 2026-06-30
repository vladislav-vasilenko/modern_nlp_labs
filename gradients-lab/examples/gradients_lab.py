from __future__ import annotations

import math
import torch
import torch.nn as nn

from gradients_lab import (
    manual_neuron_backward,
    track_gradient_norms,
    clip_gradients_by_value_,
    clip_gradients_by_norm_,
)


def report_check(name: str, diff: float, threshold: float = 1e-6) -> None:
    """Печатает разность между ручным расчетом и PyTorch и статус OK/FAIL."""
    status = "OK" if diff <= threshold else "FAIL"
    print(f"  {name:<30}: diff = {diff:.2e} (порог {threshold:.2e}) [{status}]")


def step_1_order_problem():
    print("\n" + "=" * 80)
    print("ШАГ 1: Ручное обратное распространение ошибки vs PyTorch Autograd")
    print("=" * 80)
    
    # Инициализируем входные данные
    x = torch.tensor([1.2, -0.8, 2.5])
    w = torch.tensor([0.6, 0.3, -0.4])
    b = torch.tensor(0.1)
    y = torch.tensor(1.0)
    
    print(f"Вход (x): {x.tolist()}")
    print(f"Веса (w): {w.tolist()}")
    print(f"Смещение (b): {b.item()}")
    print(f"Целевой Y: {y.item()}")
    
    # 1. Ручной расчет градиентов
    grad_x_man, grad_w_man, grad_b_man = manual_neuron_backward(x, w, b, y)
    
    # 2. Расчет с помощью PyTorch Autograd
    x_pt = x.clone().requires_grad_(True)
    w_pt = w.clone().requires_grad_(True)
    b_pt = b.clone().requires_grad_(True)
    
    z = torch.sum(x_pt * w_pt) + b_pt
    a = torch.sigmoid(z)
    loss = 0.5 * (a - y) ** 2
    loss.backward()
    
    # Сравниваем результаты
    diff_x = torch.norm(grad_x_man - x_pt.grad).item()
    diff_w = torch.norm(grad_w_man - w_pt.grad).item()
    diff_b = torch.abs(grad_b_man - b_pt.grad).item()
    
    print("\nСравнение градиентов:")
    print(f"  Ручной grad_w:  {grad_w_man.numpy()}")
    print(f"  Autograd grad_w: {w_pt.grad.numpy()}")
    print(f"  Ручной grad_b:  {grad_b_man.item():.6f}")
    print(f"  Autograd grad_b: {b_pt.grad.item():.6f}")
    
    print("\nПроверка численного совпадения:")
    report_check("Градиент по входу (x)", diff_x)
    report_check("Градиент по весам (w)", diff_w)
    report_check("Градиент по смещению (b)", diff_b)


def make_deep_mlp(depth: int, width: int, activation_fn: nn.Module, init_type: str) -> nn.Module:
    """Создает глубокий MLP с заданной инициализацией и активацией."""
    layers = []
    input_dim = width
    
    for i in range(depth):
        layer = nn.Linear(input_dim, width, bias=False)
        
        # Инициализация
        if init_type == "normal_1.0":
            nn.init.normal_(layer.weight, mean=0.0, std=1.0)
        elif init_type == "normal_0.01":
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
        elif init_type == "kaiming":
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        else:
            raise ValueError(f"Unknown init type: {init_type}")
            
        layers.append(layer)
        layers.append(activation_fn)
        
    # Добавляем финальный слой для проекции в размерность 1
    final_layer = nn.Linear(width, 1, bias=False)
    nn.init.normal_(final_layer.weight, std=0.1)
    layers.append(final_layer)
    
    return nn.Sequential(*layers)


def step_2_vanishing_exploding():
    print("\n" + "=" * 80)
    print("ШАГ 2: Затухание и взрыв градиентов в глубоких сетях")
    print("=" * 80)
    
    depth = 50
    width = 16
    
    x = torch.randn(4, width)
    y = torch.randn(4, 1)
    
    configs = [
        ("Конфигурация A (Std=1.0, Sigmoid)", nn.Sigmoid(), "normal_1.0"),
        ("Конфигурация B (Std=0.01, Sigmoid)", nn.Sigmoid(), "normal_0.01"),
        ("Конфигурация C (Kaiming, ReLU)", nn.ReLU(), "kaiming")
    ]
    
    for name, act, init in configs:
        print(f"\nТестируем: {name}")
        model = make_deep_mlp(depth, width, act, init)
        
        # Считаем нормы градиентов
        norms_info = track_gradient_norms(model, x, y)
        
        # Фильтруем веса слоев (исключаем активации, у которых нет параметров)
        linear_layer_norms = [res for res in norms_info if "weight" in res["name"]]
        
        total_layers = len(linear_layer_norms)
        print(f"Всего слоев с весами: {total_layers}")
        
        # Выбираем ключевые слои для показа:
        # первые (близкие ко входу), средние и последние (близкие к выходу)
        indices_to_show = [0, 10, 20, 30, 40, total_layers - 1]
        
        print(f"  {'Слой':<10} | {'Форма':<15} | {'Норма весов':<12} | {'Норма градиента':<18}")
        print("  " + "-" * 62)
        for idx in indices_to_show:
            res = linear_layer_norms[idx]
            # name выглядит как '0.weight', '2.weight' и т.д.
            layer_num = res["name"].split(".")[0]
            print(f"  Layer {layer_num:<5} | {str(res['shape']):<15} | {res['param_norm']:<12.4f} | {res['grad_norm']:<18.2e}")


def step_3_clipping():
    print("\n" + "=" * 80)
    print("ШАГ 3: Клиппинг градиентов (Gradient Clipping)")
    print("=" * 80)
    
    # 1. Проверяем клиппинг по значению (Value clipping)
    print("\nТест 1: Clipping by Value")
    p1 = torch.tensor([1.5, -2.0, 3.5]).requires_grad_(True)
    p2 = torch.tensor([-0.2, 0.9]).requires_grad_(True)
    
    p1.grad = torch.tensor([2.5, -1.8, 0.4])
    p2.grad = torch.tensor([-3.0, 1.2])
    
    # Reference
    p1_ref = p1.clone()
    p2_ref = p2.clone()
    p1_ref.grad = p1.grad.clone()
    p2_ref.grad = p2.grad.clone()
    
    nn.utils.clip_grad_value_([p1_ref, p2_ref], clip_value=1.0)
    
    # Custom
    clip_gradients_by_value_([p1, p2], clip_value=1.0)
    
    diff_val_1 = torch.norm(p1.grad - p1_ref.grad).item()
    diff_val_2 = torch.norm(p2.grad - p2_ref.grad).item()
    
    print(f"  Gradients before value clip: p1.grad = {p1.grad.tolist()}, p2.grad = {p2.grad.tolist()} (placeholder before action)")
    print(f"  Custom clipped gradients:   p1.grad = {p1.grad.tolist()}, p2.grad = {p2.grad.tolist()}")
    print(f"  PyTorch clipped gradients:  p1_ref.grad = {p1_ref.grad.tolist()}, p2_ref.grad = {p2_ref.grad.tolist()}")
    
    report_check("Сравнение Value Clip (p1)", diff_val_1)
    report_check("Сравнение Value Clip (p2)", diff_val_2)
    
    # 2. Проверяем клиппинг по норме (Norm clipping)
    print("\nТест 2: Clipping by Norm")
    p3 = torch.tensor([1.0, 2.0]).requires_grad_(True)
    p4 = torch.tensor([-0.5, 1.5]).requires_grad_(True)
    
    p3.grad = torch.tensor([4.0, 3.0])  # norm = 5.0
    p4.grad = torch.tensor([12.0, 5.0]) # norm = 13.0
    # Combined norm = sqrt(5.0^2 + 13.0^2) = sqrt(25 + 169) = sqrt(194) ~ 13.928
    
    p3_ref = p3.clone()
    p4_ref = p4.clone()
    p3_ref.grad = p3.grad.clone()
    p4_ref.grad = p4.grad.clone()
    
    ref_norm = nn.utils.clip_grad_norm_([p3_ref, p4_ref], max_norm=5.0)
    custom_norm = clip_gradients_by_norm_([p3, p4], max_norm=5.0)
    
    diff_norm_val = abs(ref_norm - custom_norm)
    diff_g3 = torch.norm(p3.grad - p3_ref.grad).item()
    diff_g4 = torch.norm(p4.grad - p4_ref.grad).item()
    
    print(f"  Исходная норма: {ref_norm:.4f}")
    print(f"  Суммарная норма после клиппинга (порог 5.0):")
    p3_norm_after = p3.grad.norm().item()
    p4_norm_after = p4.grad.norm().item()
    comb_norm_after = math.sqrt(p3_norm_after**2 + p4_norm_after**2)
    print(f"    Custom: {comb_norm_after:.4f}")
    
    report_check("Сравнение возвращаемой исходной нормы", diff_norm_val)
    report_check("Сравнение Norm Clip (p3)", diff_g3)
    report_check("Сравнение Norm Clip (p4)", diff_g4)


def main():
    print("=" * 80)
    print("ЗАПУСК ЛАБОРАТОРНОЙ РАБОТЫ ПО ГРАДИЕНТАМ")
    print("=" * 80)
    
    step_1_order_problem()
    step_2_vanishing_exploding()
    step_3_clipping()
    
    print("\n" + "=" * 80)
    print("Лабораторный скрипт завершил работу.")
    print("=" * 80)


if __name__ == "__main__":
    main()
