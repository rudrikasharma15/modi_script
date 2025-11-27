#!/usr/bin/env python3
"""
Get step-by-step layer outputs with visualization
"""
import torch
import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def get_all_layer_outputs_visual(model, image_path, output_dir="layer_outputs"):
    """Get and visualize every single layer output"""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Storage for outputs
    layer_outputs = {}
    layer_shapes = {}
    
    # Hook function to capture outputs
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                layer_outputs[name] = output.detach().cpu()
                layer_shapes[name] = output.shape
                print(f"✓ {name}: {output.shape}")
        return hook
    
    # Register hooks on ALL layers
    hooks = []
    for idx, (name, module) in enumerate(model.model.named_modules()):
        if len(list(module.children())) == 0:  # Only leaf modules
            hook = module.register_forward_hook(make_hook(f"Layer_{idx}_{name}"))
            hooks.append(hook)
    
    # Run inference
    print(f"\n🔍 Processing: {image_path}")
    print("=" * 80)
    
    with torch.no_grad():
        results = model(image_path)
    
    print("=" * 80)
    print(f"✅ Captured {len(layer_outputs)} layer outputs\n")
    
    # Visualize each layer
    print("📊 Creating visualizations...")
    for layer_name, activation in layer_outputs.items():
        if len(activation.shape) == 4:  # (batch, channels, height, width)
            visualize_layer(activation, layer_name, output_dir)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return layer_outputs, layer_shapes

def visualize_layer(activation, layer_name, output_dir):
    """Create visualization for a single layer"""
    
    # Get first image, take mean across channels or show individual channels
    act = activation[0].numpy()  # Shape: (channels, height, width)
    
    # Save different views:
    
    # 1. Mean activation (average across all channels)
    mean_act = act.mean(axis=0)
    plt.figure(figsize=(10, 8))
    plt.imshow(mean_act, cmap='jet')
    plt.colorbar(label='Activation Strength')
    plt.title(f'{layer_name}\nMean Activation (averaged across channels)')
    plt.axis('off')
    safe_name = layer_name.replace('/', '_').replace('.', '_')
    plt.savefig(f"{output_dir}/{safe_name}_mean.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. First 16 individual channels
    if act.shape[0] >= 1:
        n_channels = min(16, act.shape[0])
        fig, axes = plt.subplots(4, 4, figsize=(15, 15))
        fig.suptitle(f'{layer_name} - Individual Channels', fontsize=14)
        
        for i in range(16):
            ax = axes[i // 4, i % 4]
            if i < act.shape[0]:
                im = ax.imshow(act[i], cmap='viridis')
                ax.set_title(f'Channel {i}', fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{safe_name}_channels.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # 3. Max activation (strongest channel at each pixel)
    max_act = act.max(axis=0)
    plt.figure(figsize=(10, 8))
    plt.imshow(max_act, cmap='hot')
    plt.colorbar(label='Max Activation')
    plt.title(f'{layer_name}\nMax Activation (strongest response)')
    plt.axis('off')
    plt.savefig(f"{output_dir}/{safe_name}_max.png", dpi=150, bbox_inches='tight')
    plt.close()

def create_summary_report(layer_shapes, output_dir):
    """Create a text summary of all layers"""
    
    with open(f"{output_dir}/layer_summary.txt", 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("YOLO MODEL LAYER-BY-LAYER SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        for layer_name, shape in layer_shapes.items():
            f.write(f"Layer: {layer_name}\n")
            f.write(f"  Shape: {shape}\n")
            if len(shape) == 4:
                f.write(f"  Channels: {shape[1]}, Size: {shape[2]}x{shape[3]}\n")
            f.write("\n")

# MAIN EXECUTION
if __name__ == "__main__":
    # Load model
    model = YOLO("runs/modi_matra/train_full_7k2/weights/best.pt")

    
    # Process single image
    image_path = "modi_full_7k/merged/images/test/WhatsApp Image 2021-08-19 at 5.37.46 PM_82.png"
    
    layer_outputs, layer_shapes = get_all_layer_outputs_visual(
        model, 
        image_path,
        output_dir="layer_visualizations"
    )
    
    # Create summary
    create_summary_report(layer_shapes, "layer_visualizations")
    
    print("\n✅ Done! Check the 'layer_visualizations' folder")
    print("   - *_mean.png: Average activation across channels")
    print("   - *_channels.png: Individual channel activations")
    print("   - *_max.png: Maximum activation at each pixel")
    print("   - layer_summary.txt: Text summary of all layers")