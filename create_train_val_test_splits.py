#!/usr/bin/env python3
"""
Create train/val/test splits with truly unseen test set.
Train: 70%, Val: 15%, Test: 15% (completely held out)
"""
import argparse
import shutil
import random
from pathlib import Path
from collections import Counter
import yaml

def create_splits(dataset_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """Create train/val/test splits with stratification by class."""
    
    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / 'images'
    labels_dir = dataset_path / 'labels'
    
    # Verify directories exist
    if not images_dir.exists() or not labels_dir.exists():
        print(f"❌ Error: {images_dir} or {labels_dir} not found!")
        return
    
    # Find all labeled images
    label_files = list(labels_dir.glob('*.txt'))
    print(f"📊 Found {len(label_files)} labeled images")
    
    # Group by primary class (for stratified split)
    class_groups = {0: [], 1: [], 2: []}  # top, side, bottom
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            if lines:
                # Get primary class (most common in image)
                classes = [int(line.split()[0]) for line in lines if line.strip()]
                if classes:
                    primary_class = Counter(classes).most_common(1)[0][0]
                    class_groups[primary_class].append(label_file.stem)
    
    print(f"\n📈 Class distribution:")
    for cls, stems in class_groups.items():
        class_names = {0: "top_matra", 1: "side_matra", 2: "bottom_matra"}
        print(f"  {class_names[cls]}: {len(stems)} images")
    
    # Shuffle each class group
    for cls in class_groups:
        random.shuffle(class_groups[cls])
    
    # Split each class proportionally
    train_stems = []
    val_stems = []
    test_stems = []
    
    for cls, stems in class_groups.items():
        n = len(stems)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_stems.extend(stems[:n_train])
        val_stems.extend(stems[n_train:n_train+n_val])
        test_stems.extend(stems[n_train+n_val:])
    
    # Shuffle combined lists
    random.shuffle(train_stems)
    random.shuffle(val_stems)
    random.shuffle(test_stems)
    
    print(f"\n✂️  Split sizes:")
    print(f"  Train: {len(train_stems)} ({len(train_stems)/len(label_files)*100:.1f}%)")
    print(f"  Val:   {len(val_stems)} ({len(val_stems)/len(label_files)*100:.1f}%)")
    print(f"  Test:  {len(test_stems)} ({len(test_stems)/len(label_files)*100:.1f}%)")
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        (dataset_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dataset_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Move files to split directories
    def move_to_split(stems, split_name):
        for stem in stems:
            # Find image file
            img_file = None
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                candidate = images_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_file = candidate
                    break
            
            if img_file:
                # Move image
                dst_img = dataset_path / 'images' / split_name / img_file.name
                shutil.move(str(img_file), str(dst_img))
                
                # Move label
                label_file = labels_dir / f"{stem}.txt"
                if label_file.exists():
                    dst_label = dataset_path / 'labels' / split_name / f"{stem}.txt"
                    shutil.move(str(label_file), str(dst_label))
    
    print("\n📦 Moving files to splits...")
    move_to_split(train_stems, 'train')
    move_to_split(val_stems, 'val')
    move_to_split(test_stems, 'test')
    
    # Clean up empty root directories
    try:
        if images_dir.exists() and not list(images_dir.glob('*.*')):
            pass  # Leave it for now
        if labels_dir.exists() and not list(labels_dir.glob('*.txt')):
            pass  # Leave it for now
    except:
        pass
    
    # Create YAML config for YOLO
    yaml_config = {
        'path': str(dataset_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {
            0: 'top_matra',
            1: 'side_matra',
            2: 'bottom_matra'
        }
    }
    
    yaml_path = dataset_path / 'modi_matra.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False)
    
    print(f"\n✅ Dataset splits created!")
    print(f"📄 Config saved: {yaml_path}")
    
    # Create metadata file
    metadata = {
        'total_images': len(label_files),
        'train': len(train_stems),
        'val': len(val_stems),
        'test': len(test_stems),
        'class_distribution': {
            'top_matra': len(class_groups[0]),
            'side_matra': len(class_groups[1]),
            'bottom_matra': len(class_groups[2])
        }
    }
    
    import json
    metadata_path = dataset_path / 'split_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📄 Metadata saved: {metadata_path}")
    
    print("\n" + "="*70)
    print("🎯 DATASET READY FOR TRAINING!")
    print("="*70)
    print(f"\n📊 Final Structure:")
    print(f"  {dataset_path}/")
    print(f"  ├── images/")
    print(f"  │   ├── train/ ({len(train_stems)} images) - For training")
    print(f"  │   ├── val/   ({len(val_stems)} images) - For validation during training")
    print(f"  │   └── test/  ({len(test_stems)} images) - UNSEEN data for final evaluation")
    print(f"  ├── labels/")
    print(f"  │   ├── train/")
    print(f"  │   ├── val/")
    print(f"  │   └── test/")
    print(f"  └── modi_matra.yaml")
    
    print("\n📝 Next Steps:")
    print(f"  1. Train model:")
    print(f"     python train_modi_matra.py --data {yaml_path} --epochs 200 --batch 32")
    print(f"\n  2. Evaluate on UNSEEN test set:")
    print(f"     python evaluate_unseen.py --model runs/.../best.pt --test_dir {dataset_path}")
    print("="*70 + "\n")
    
    print("⚠️  IMPORTANT: The test/ directory contains UNSEEN data.")
    print("   DO NOT use it during training - only for final evaluation!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create train/val/test splits')
    parser.add_argument('--dataset', required=True, help='Dataset directory')
    parser.add_argument('--train_ratio', type=float, default=0.70, help='Train ratio (default: 0.70)')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Val ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Verify ratios sum to 1.0
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 0.01:
        print(f"❌ Error: Ratios must sum to 1.0 (got {total})")
        exit(1)
    
    create_splits(args.dataset, args.train_ratio, args.val_ratio, args.test_ratio)