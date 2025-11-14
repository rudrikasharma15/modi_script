#!/usr/bin/env python3
"""
train_modi_matra.py
Training script for Modi matra detection using YOLOv8.

Usage:
    # Basic training
    python train_modi_matra.py --data datasets/modi_matra_sample/modi_matra.yaml
    
    # Advanced training
    python train_modi_matra.py --data datasets/modi_matra_sample/modi_matra.yaml --epochs 150 --batch 32 --imgsz 800
"""
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch

def train_model(data_yaml, 
                model='yolov8n.pt',
                epochs=100, 
                imgsz=640, 
                batch=16,
                patience=20,
                device='',
                project='runs/modi_matra',
                name='train'):
    """
    Train YOLO model for Modi matra detection.
    
    Args:
        data_yaml: Path to dataset YAML file
        model: Model to use (yolov8n.pt, yolov8s.pt, etc.)
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        patience: Early stopping patience
        device: Device to use ('' for auto, 'cpu', '0', etc.)
        project: Project directory
        name: Experiment name
    """
    
    # Check if CUDA is available
    if device == '' and torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    elif device == '':
        print("⚠ Training on CPU (will be slow)")
    
    # Load model
    print(f"\nLoading model: {model}")
    model = YOLO(model)
    
    # Print dataset info
    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    
    print(f"Dataset: {data_yaml}")
    
    # Training parameters
    print(f"\nTraining configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Patience: {patience}")
    print(f"  Device: {device if device else 'auto'}")
    
    # Train
    print(f"\n{'='*70}")
    print("Starting training...")
    print(f"{'='*70}\n")
    
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        device=device,
        project=project,
        name=name,
        # Augmentation
        hsv_h=0.015,      # Image HSV-Hue augmentation
        hsv_s=0.7,        # Image HSV-Saturation augmentation  
        hsv_v=0.4,        # Image HSV-Value augmentation
        degrees=10.0,     # Image rotation
        translate=0.1,    # Image translation
        scale=0.5,        # Image scale
        shear=0.0,        # Image shear
        perspective=0.0,  # Image perspective
        flipud=0.0,       # Vertical flip (no flip for script)
        fliplr=0.0,       # Horizontal flip (no flip for script)
        mosaic=1.0,       # Mosaic augmentation
        mixup=0.0,        # MixUp augmentation
        # Training settings
        optimizer='auto', # Optimizer (SGD, Adam, AdamW, auto)
        lr0=0.01,         # Initial learning rate
        lrf=0.01,         # Final learning rate
        momentum=0.937,   # Momentum
        weight_decay=0.0005,  # Weight decay
        warmup_epochs=3.0,    # Warmup epochs
        warmup_momentum=0.8,  # Warmup momentum
        warmup_bias_lr=0.1,   # Warmup bias learning rate
        box=7.5,          # Box loss gain
        cls=0.5,          # Classification loss gain
        dfl=1.5,          # Distribution focal loss gain
        # Validation
        val=True,         # Validate during training
        plots=True,       # Save plots
        save=True,        # Save checkpoints
        save_period=-1,   # Save checkpoint every N epochs (-1 to disable)
        cache=False,      # Cache images (True, False, or 'disk')
        exist_ok=False,   # Overwrite existing experiment
        verbose=True,     # Verbose output
    )
    
    print(f"\n{'='*70}")
    print("Training completed!")
    print(f"{'='*70}")
    
    # Print results location
    save_dir = Path(project) / name
    print(f"\nResults saved to: {save_dir.absolute()}")
    print(f"Best model: {save_dir / 'weights' / 'best.pt'}")
    print(f"Last model: {save_dir / 'weights' / 'last.pt'}")
    
    # Print key metrics
    if results:
        print("\nFinal metrics:")
        print(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    
    print(f"\n{'='*70}")
    print("Next steps:")
    print(f"{'='*70}")
    print(f"\n1. Validate model:")
    print(f"   yolo detect val model={save_dir / 'weights' / 'best.pt'} data={data_yaml}")
    print(f"\n2. Test predictions:")
    print(f"   python predict_modi_matra.py --model {save_dir / 'weights' / 'best.pt'} --source test_images/")
    print(f"\n3. View training plots:")
    print(f"   open {save_dir / 'results.png'}")
    print(f"{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(
        description='Train YOLOv8 for Modi matra detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick training with small model
  python train_modi_matra.py --data datasets/modi_sample/modi_matra.yaml --epochs 50
  
  # Full training with larger model
  python train_modi_matra.py --data datasets/modi_full/modi_matra.yaml --model yolov8s.pt --epochs 150 --batch 32
  
  # Resume training
  python train_modi_matra.py --data datasets/modi_sample/modi_matra.yaml --model runs/modi_matra/train/weights/last.pt

Models:
  yolov8n.pt - Nano (fastest, least accurate)
  yolov8s.pt - Small
  yolov8m.pt - Medium
  yolov8l.pt - Large
  yolov8x.pt - Extra large (slowest, most accurate)
        """
    )
    
    # Required arguments
    parser.add_argument('--data', required=True, 
                       help='Path to dataset YAML file')
    
    # Model arguments
    parser.add_argument('--model', default='yolov8n.pt',
                       help='Model to use (default: yolov8n.pt)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size (default: 640)')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience (default: 20)')
    parser.add_argument('--device', default='',
                       help='Device (default: auto, options: cpu, 0, 0,1, etc.)')
    
    # Output arguments
    parser.add_argument('--project', default='runs/modi_matra',
                       help='Project directory (default: runs/modi_matra)')
    parser.add_argument('--name', default='train',
                       help='Experiment name (default: train)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.data).exists():
        print(f"ERROR: Dataset YAML not found: {args.data}")
        return
    
    # Train
    train_model(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name
    )

if __name__ == '__main__':
    main()
