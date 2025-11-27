#!/usr/bin/env python3
"""
Visualize ALL major stages of YOLO model
Shows one representative layer from each stage
"""
import torch
import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

def visualize_layer(activation, layer_name, output_dir, image_name, stage_name):
    """Create visualization for a single layer"""
    
    if len(activation.shape) != 4:
        print(f"Skipping {layer_name} - not a spatial tensor (shape: {activation.shape})")
        return
    
    act = activation[0].cpu().numpy()  # Shape: (channels, height, width)
    
    safe_layer = layer_name.replace('/', '_').replace('.', '_')
    safe_img = Path(image_name).stem
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{stage_name}\n{layer_name}\nImage: {safe_img}', fontsize=14, fontweight='bold')
    
    # 1. Mean activation
    mean_act = act.mean(axis=0)
    im1 = axes[0].imshow(mean_act, cmap='jet')
    axes[0].set_title('Mean Activation')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    # 2. Max activation
    max_act = act.max(axis=0)
    im2 = axes[1].imshow(max_act, cmap='hot')
    axes[1].set_title('Max Activation')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    # 3. First 9 channels
    if act.shape[0] >= 9:
        channels_grid = np.zeros((3*act.shape[1], 3*act.shape[2]))
        for i in range(9):
            row = i // 3
            col = i % 3
            channels_grid[row*act.shape[1]:(row+1)*act.shape[1], 
                         col*act.shape[2]:(col+1)*act.shape[2]] = act[i]
        im3 = axes[2].imshow(channels_grid, cmap='viridis')
        axes[2].set_title('First 9 Channels')
    else:
        im3 = axes[2].imshow(act[0], cmap='viridis')
        axes[2].set_title(f'Channel 0')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{safe_img}_STAGE_{stage_name}_{safe_layer}.png", 
                dpi=150, bbox_inches='tight')
    plt.close()

# Replace the "process_all_images" call with a single-image version

def process_single_image(model_path, image_path, output_dir):
    """Process one image and visualize all YOLO stages"""
    
    # Load model
    print("Loading YOLO model...")
    model = YOLO(model_path)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Define YOLO stages (same as before)
    YOLO_STAGES = {
        'STAGE_01_INPUT_CONV': 'model.0.conv',
        'STAGE_02_STEM': 'model.1.conv',
        'STAGE_03_BACKBONE_L1': 'model.2.cv1.conv',
        'STAGE_04_BACKBONE_L2': 'model.4.cv1.conv',
        'STAGE_05_BACKBONE_L3': 'model.6.cv1.conv',
        'STAGE_06_BOTTLENECK': 'model.8.cv1.conv',
        'STAGE_07_SPPF': 'model.9.cv2.conv',
        'STAGE_08_UPSAMPLE_1': 'model.10',
        'STAGE_09_CONCAT_1': 'model.11',
        'STAGE_10_NECK_L1': 'model.12.cv1.conv',
        'STAGE_11_UPSAMPLE_2': 'model.13',
        'STAGE_12_CONCAT_2': 'model.14',
        'STAGE_13_NECK_L2': 'model.15.cv1.conv',
        'STAGE_14_DOWNSAMPLE_1': 'model.16.conv',
        'STAGE_15_CONCAT_3': 'model.17',
        'STAGE_16_HEAD_MEDIUM': 'model.18.cv1.conv',
        'STAGE_17_DOWNSAMPLE_2': 'model.19.conv',
        'STAGE_18_CONCAT_4': 'model.20',
        'STAGE_19_HEAD_LARGE': 'model.21.cv1.conv',
        'STAGE_20_DETECT_SMALL_BBOX': 'model.22.cv2.0.2',
        'STAGE_21_DETECT_SMALL_CLASS': 'model.22.cv3.0.2',
        'STAGE_22_DETECT_MEDIUM_BBOX': 'model.22.cv2.1.2',
        'STAGE_23_DETECT_MEDIUM_CLASS': 'model.22.cv3.1.2',
        'STAGE_24_DETECT_LARGE_BBOX': 'model.22.cv2.2.2',
        'STAGE_25_DETECT_LARGE_CLASS': 'model.22.cv3.2.2',
    }
    
    layer_outputs = {}
    
    # Hook function
    def make_hook(name, stage_name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                layer_outputs[name] = (output.detach().cpu(), stage_name)
                print(f"  ✓ {stage_name}: {name} -> {output.shape}")
        return hook
    
    # Register hooks
    hooks = []
    for stage_name, layer_name in YOLO_STAGES.items():
        module = model.model
        for part in layer_name.split('.'):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        hook = module.register_forward_hook(make_hook(layer_name, stage_name))
        hooks.append(hook)
    
    # Run inference
    with torch.no_grad():
        results = model(str(image_path))
    
    # Visualize each stage
    for layer_name, (activation, stage_name) in layer_outputs.items():
        visualize_layer(activation, layer_name, output_dir, Path(image_path).name, stage_name)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    print(f"\n✅ Completed visualization for {Path(image_path).name}")
    create_summary(output_dir, YOLO_STAGES)



def create_summary(output_dir, stages):
    """Create a summary document explaining all stages"""
    
    summary = """
================================================================================
YOLO MODEL - ALL STAGES EXPLAINED
================================================================================

BACKBONE (Feature Extraction - Stages 1-6):
-------------------------------------------
STAGE_01_INPUT_CONV      : Initial convolution - processes raw image
STAGE_02_STEM            : Stem block - first downsampling
STAGE_03_BACKBONE_L1     : Backbone Level 1 - extracts low-level features
STAGE_04_BACKBONE_L2     : Backbone Level 2 - extracts mid-level features
STAGE_05_BACKBONE_L3     : Backbone Level 3 - extracts high-level features
STAGE_06_BOTTLENECK      : Bottleneck - deepest, most abstract features

NECK (Feature Pyramid Network - Stages 7-13):
----------------------------------------------
STAGE_07_SPPF            : Spatial Pyramid Pooling - multi-scale context
STAGE_08_UPSAMPLE_1      : First upsample - increase resolution
STAGE_09_CONCAT_1        : Concatenate with backbone features
STAGE_10_NECK_L1         : Process combined features
STAGE_11_UPSAMPLE_2      : Second upsample - further increase resolution
STAGE_12_CONCAT_2        : Concatenate with earlier features
STAGE_13_NECK_L2         : Process for small object detection

HEAD (Multi-scale Detection - Stages 14-19):
---------------------------------------------
STAGE_14_DOWNSAMPLE_1    : Downsample for medium objects
STAGE_15_CONCAT_3        : Concatenate features for medium scale
STAGE_16_HEAD_MEDIUM     : Process medium object features
STAGE_17_DOWNSAMPLE_2    : Downsample for large objects
STAGE_18_CONCAT_4        : Concatenate features for large scale
STAGE_19_HEAD_LARGE      : Process large object features

DETECTION HEADS (Final Predictions - Stages 20-25):
----------------------------------------------------
STAGE_20_DETECT_SMALL_BBOX    : Predict bounding boxes for SMALL objects (80x76 grid)
STAGE_21_DETECT_SMALL_CLASS   : Predict classes for SMALL objects
STAGE_22_DETECT_MEDIUM_BBOX   : Predict bounding boxes for MEDIUM objects (40x38 grid)
STAGE_23_DETECT_MEDIUM_CLASS  : Predict classes for MEDIUM objects
STAGE_24_DETECT_LARGE_BBOX    : Predict bounding boxes for LARGE objects (20x19 grid)
STAGE_25_DETECT_LARGE_CLASS   : Predict classes for LARGE objects

================================================================================
HOW TO READ THE VISUALIZATIONS:
================================================================================

Each stage produces 3 images:

1. MEAN ACTIVATION (left):
   - Shows average response across all channels
   - Bright areas = model is paying attention
   - Dark areas = model ignores this region

2. MAX ACTIVATION (middle):
   - Shows strongest response from any channel
   - Highlights the most important features detected

3. CHANNEL GRID (right):
   - Shows first 9 individual feature channels
   - Each channel learns different patterns

================================================================================
WHAT TO LOOK FOR:
================================================================================

✅ HEALTHY MODEL:
- Stages 1-6: Progressive abstraction (edges → shapes → objects)
- Stages 7-13: Clear multi-scale features
- Stages 14-19: Focused on object regions
- Stages 20-25: Grid patterns with activations at object locations

❌ PROBLEMATIC PATTERNS:
- All black/white images = dead/exploding neurons
- No progression = model not learning
- Empty detection heads = no objects detected

================================================================================
"""
    
    with open(f"{output_dir}/README_STAGES.txt", 'w') as f:
        f.write(summary)
    
    print(f"\n📄 Summary created: {output_dir}/README_STAGES.txt")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    MODEL_PATH = "runs/modi_matra/train_full_7k2/weights/best.pt"
    IMAGE_PATH = "modi_full_7k/merged/images/test/sample_image.png"  # <-- single image
    OUTPUT_DIR = "yolo_single_image_visualization"

    # process_single_image(MODEL_PATH, IMAGE_PATH, OUTPUT_DIR)

    
    print("""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                                                                        ║
    ║         YOLO ALL STAGES VISUALIZATION                                  ║
    ║         Visualizes ALL 25+ stages of YOLO pipeline                     ║
    ║                                                                        ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """)
    
    process_all_images(MODEL_PATH, IMAGES_DIR, OUTPUT_DIR)
    
    print("""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                                                                        ║
    ║  ✅ COMPLETE! Check the output folder for:                             ║
    ║     - Stage visualizations for each image                              ║
    ║     - README_STAGES.txt (explanation of all stages)                    ║
    ║                                                                        ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """)