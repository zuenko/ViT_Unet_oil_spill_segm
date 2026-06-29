"""
Compute model complexity: Parameters, FLOPs, Memory, Inference Speed
Uses thop and torchsummary for accurate calculations.
"""
import os
import sys
import time
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import (
    ViTUNet, UNetBaseline, DeepLabV3Plus, CBDNet,
    TransUNet, DSUNet, LRAUNet, DAENet, SegFormerWrapper,
    ViTUNetDeepSkip
)

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_speed(model, input_size=(1, 3, 224, 224), device='cuda', num_runs=100):
    """Measure inference FPS."""
    model = model.to(device)
    model.eval()
    
    # Warmup
    dummy_input = torch.randn(input_size).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
    
    elapsed = time.time() - start_time
    fps = num_runs / elapsed
    
    return fps

def measure_memory(model, input_size=(1, 3, 224, 224), device='cuda'):
    """Measure GPU memory usage."""
    if device != 'cuda':
        return 0
    
    model = model.to(device)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    dummy_input = torch.randn(input_size).to(device)
    
    with torch.no_grad():
        _ = model(dummy_input)
    
    memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return memory_mb

def compute_flops_thop(model, input_size=(1, 3, 224, 224)):
    """Compute FLOPs using thop. Falls back to params-only if thop fails on a
    given architecture (e.g., timm ViT modules, custom forwards)."""
    fallback_params_m = count_parameters(model) / 1e6
    try:
        from thop import profile
    except ImportError:
        print("Warning: thop not installed. Install with: pip install thop")
        return 0.0, fallback_params_m
    # thop traces the forward pass with BN/dropout, so model must be in eval()
    # mode (otherwise BN with batch_size=1 raises "Expected more than 1 value").
    model.eval()
    dummy_input = torch.randn(input_size)
    result = (float('nan'), fallback_params_m)
    try:
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        result = (flops / 1e9, params / 1e6)
    except Exception as e:
        print(f"  [thop failed for this model: {type(e).__name__}: {str(e)[:80]} — reporting params only]")
    finally:
        # thop leaves total_ops/total_params attrs and forward hooks on every
        # module; for some timm-based architectures these break subsequent
        # forward passes used for memory/FPS measurements. Always strip them.
        for m in model.modules():
            for attr in ('total_ops', 'total_params'):
                if hasattr(m, attr):
                    try:
                        delattr(m, attr)
                    except Exception:
                        pass
            try:
                m._forward_hooks.clear()
            except Exception:
                pass
    return result

def analyze_all_models():
    """Analyze all models and create comparison table."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # All 9 main architectures evaluated in the paper, plus the deep-skip ablation variant.
    # Each entry: (display_name, model_instance, returns_tuple_for_boundary)
    models_config = [
        ('ViT-UNet',             ViTUNet(model_name='vit_small_patch16_224', pretrained=False), False),
        ('U-Net',                UNetBaseline(),                                                False),
        ('DeepLabV3+',           DeepLabV3Plus(pretrained=False),                               False),
        ('CBD-Net',              CBDNet(pretrained=False),                                      True),
        ('TransUNet',            TransUNet(pretrained=False),                                   False),
        ('DS-UNet',              DSUNet(),                                                      True),
        ('LRA-UNet',             LRAUNet(),                                                     False),
        ('DAENet',               DAENet(pretrained=False),                                      False),
        ('SegFormer',            SegFormerWrapper(pretrained=False),                            False),
        ('ViT-UNet (Deep Skip)', ViTUNetDeepSkip(pretrained=False),                             False),
    ]
    
    results = []
    
    print("="*100)
    print("COMPUTATIONAL COMPLEXITY ANALYSIS")
    print("="*100)
    print(f"{'Model':<25} {'Params (M)':<15} {'FLOPs (G)':<15} {'Memory (MB)':<15} {'FPS':<10}")
    print("-"*100)
    
    for name, model, use_boundary in models_config:
        try:
            # Count parameters
            params = count_parameters(model)
            
            # Compute FLOPs
            flops, _ = compute_flops_thop(model)
            
            # Measure memory
            memory = measure_memory(model, device=device) if device == 'cuda' else 0
            
            # Measure speed
            fps = measure_inference_speed(model, device=device, num_runs=50)
            
            results.append({
                'Model': name,
                'Parameters (M)': f"{params/1e6:.1f}",
                'FLOPs (G)': f"{flops:.1f}",
                'Memory (MB)': f"{memory:.0f}",
                'FPS': f"{fps:.0f}"
            })
            
            print(f"{name:<25} {params/1e6:<15.1f} {flops:<15.1f} {memory:<15.0f} {fps:<10.0f}")
            
        except Exception as e:
            print(f"{name:<25} ERROR: {str(e)[:50]}")
    
    print("="*100)
    
    # Save to CSV
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('models_results/model_complexity_analysis.csv', index=False)
    print("\nResults saved to: models_results/model_complexity_analysis.csv")
    
    return results

if __name__ == "__main__":
    os.makedirs('models_results', exist_ok=True)
    analyze_all_models()
