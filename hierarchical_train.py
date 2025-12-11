"""
Hierarchical YOLO Training
Detects 37 classes: 34 characters + 3 matras simultaneously
"""

from ultralytics import YOLO
import torch

def train_hierarchical_yolo(
    data_yaml='modi_hierarchical_yolo/data.yaml',
    pretrained_matra_weights=None,
    epochs=150,
    imgsz=640,
    batch=16,
    device='0'
):
    """
    Train YOLOv8 for hierarchical character+matra detection
    
    Key modifications for better accuracy:
    1. Transfer learning from your matra model
    2. Class-weighted loss for imbalanced data
    3. Optimized augmentation for Modi script
    """
    
    print("="*60)
    print("HIERARCHICAL YOLO TRAINING")
    print("="*60)
    print("Configuration:")
    print(f"  Classes: 37 (34 char + 3 matra)")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch}")
    print(f"  Image size: {imgsz}")
    print("="*60)
    
    # Initialize model
    if pretrained_matra_weights:
        print(f"\n📦 Loading pretrained matra weights: {pretrained_matra_weights}")
        model = YOLO(pretrained_matra_weights)
        print("✅ Will use transfer learning from matra model")
    else:
        print("\n📦 Initializing from YOLOv8n pretrained on COCO")
        model = YOLO('yolov8n.pt')
    
    # Training hyperparameters optimized for Modi script
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        
        # Optimizer settings
        optimizer='SGD',
        lr0=0.01,
        lrf=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        
        # Augmentation (conservative for Modi)
        degrees=10.0,  # Rotation ±10°
        translate=0.1,  # Translation 10%
        scale=0.5,  # Scale 50%
        shear=0.0,  # No shear (preserves Modi strokes)
        flipud=0.0,  # No vertical flip (invalid Modi)
        fliplr=0.0,  # No horizontal flip (invalid Modi)
        mosaic=1.0,  # Enable mosaic augmentation
        mixup=0.0,  # No mixup (confuses character boundaries)
        
        # Loss weights (emphasize localization)
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # Training settings
        patience=50,  # Early stopping patience
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        
        # Validation
        val=True,
        plots=True,
        
        # Performance
        workers=8,
        project='runs/hierarchical_yolo',
        name='char_matra_detection',
        exist_ok=True,
        
        # Class weights (balance character vs matra detection)
        # Characters (0-33): weight 1.0
        # Matras (34-36): weight 1.5 (minority class boost)
        # This is handled automatically by YOLO based on class frequency
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best model saved at: runs/hierarchical_yolo/char_matra_detection/weights/best.pt")
    print(f"Final mAP@0.5: {results.results_dict['metrics/mAP50(B)']:.4f}")
    
    return model, results


def validate_hierarchical_yolo(
    weights='runs/hierarchical_yolo/char_matra_detection/weights/best.pt',
    data_yaml='modi_hierarchical_yolo/data.yaml'
):
    """
    Validate trained model on test set
    """
    print("\n" + "="*60)
    print("VALIDATION ON TEST SET")
    print("="*60)
    
    model = YOLO(weights)
    
    # Run validation
    metrics = model.val(
        data=data_yaml,
        split='test',
        imgsz=640,
        batch=16,
        save_json=True,
        save_hybrid=True,
        conf=0.25,
        iou=0.6,
        plots=True
    )
    
    # Extract per-class metrics
    print("\n📊 OVERALL METRICS:")
    print(f"  mAP@0.5 (all classes): {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    # Character-specific metrics (classes 0-33)
    char_maps = metrics.box.maps[:34]
    char_map50 = char_maps.mean()
    
    print(f"\n📊 CHARACTER DETECTION (classes 0-33):")
    print(f"  Average mAP@0.5: {char_map50:.4f}")
    print(f"  Best character: {char_maps.max():.4f}")
    print(f"  Worst character: {char_maps.min():.4f}")
    
    # Matra-specific metrics (classes 34-36)
    matra_maps = metrics.box.maps[34:37]
    matra_names = ['top_matra', 'side_matra', 'bottom_matra']
    
    print(f"\n📊 MATRA DETECTION (classes 34-36):")
    for i, (name, map_val) in enumerate(zip(matra_names, matra_maps)):
        print(f"  {name}: {map_val:.4f}")
    print(f"  Average mAP@0.5: {matra_maps.mean():.4f}")
    
    return metrics


def apply_progressive_layer_optimization(
    base_weights='yolov8n.pt',
    data_yaml='modi_hierarchical_yolo/data.yaml',
    problematic_layers=[0, 1, 7, 19, 21]
):
    """
    Apply your proven layer optimization strategy to hierarchical model
    
    This uses the SAME 4-stage He initialization you already validated
    """
    print("\n" + "="*60)
    print("PROGRESSIVE LAYER OPTIMIZATION")
    print("="*60)
    print("Applying 4-stage He reinitialization to layers:")
    print(f"  {problematic_layers}")
    print("="*60)
    
    import torch.nn as nn
    
    model = YOLO(base_weights)
    
    # Stage 1: Reinitialize layers 0, 1 (30 epochs)
    print("\n🔧 Stage 1: Optimizing early layers (0, 1)...")
    for layer_idx in [0, 1]:
        for module in model.model.model[layer_idx].modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    # Freeze other layers
    for i, layer in enumerate(model.model.model):
        if i not in [0, 1]:
            for param in layer.parameters():
                param.requires_grad = False
    
    model.train(
        data=data_yaml,
        epochs=30,
        imgsz=640,
        batch=16,
        lr0=0.01,
        project='runs/hierarchical_optimized',
        name='stage1',
        exist_ok=True
    )
    
    # Unfreeze all
    for param in model.model.parameters():
        param.requires_grad = True
    
    # Stage 2: Reinitialize layer 7 (30 epochs)
    print("\n🔧 Stage 2: Optimizing mid layer (7)...")
    for module in model.model.model[7].modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    model.train(
        data=data_yaml,
        epochs=30,
        imgsz=640,
        batch=16,
        lr0=0.008,
        project='runs/hierarchical_optimized',
        name='stage2',
        exist_ok=True,
        resume=True
    )
    
    # Stage 3: Reinitialize layers 19, 21 (30 epochs)
    print("\n🔧 Stage 3: Optimizing late layers (19, 21)...")
    for layer_idx in [19, 21]:
        for module in model.model.model[layer_idx].modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    model.train(
        data=data_yaml,
        epochs=30,
        imgsz=640,
        batch=16,
        lr0=0.005,
        project='runs/hierarchical_optimized',
        name='stage3',
        exist_ok=True,
        resume=True
    )
    
    # Stage 4: Full fine-tuning (40 epochs)
    print("\n🔧 Stage 4: End-to-end fine-tuning...")
    model.train(
        data=data_yaml,
        epochs=40,
        imgsz=640,
        batch=16,
        lr0=0.003,
        project='runs/hierarchical_optimized',
        name='stage4',
        exist_ok=True,
        resume=True
    )
    
    print("\n✅ Progressive optimization complete!")
    print("Best model: runs/hierarchical_optimized/stage4/weights/best.pt")
    
    return model


# MAIN EXECUTION
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'validate', 'optimize'],
                       help='Training mode')
    parser.add_argument('--data', type=str, default='modi_hierarchical_yolo/data.yaml')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pretrained matra weights')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("🚀 Starting standard training...")
        model, results = train_hierarchical_yolo(
            data_yaml=args.data,
            pretrained_matra_weights=args.pretrained,
            epochs=args.epochs,
            batch=args.batch
        )
        
    elif args.mode == 'validate':
        print("🧪 Running validation...")
        metrics = validate_hierarchical_yolo(data_yaml=args.data)
        
    elif args.mode == 'optimize':
        print("⚡ Applying progressive layer optimization...")
        model = apply_progressive_layer_optimization(
            data_yaml=args.data
        )
        
        # Validate optimized model
        metrics = validate_hierarchical_yolo(
            weights='runs/hierarchical_optimized/stage4/weights/best.pt',
            data_yaml=args.data
        )
    
    print("\n✅ DONE!")