#!/usr/bin/env python3
"""
create_splits.py
Creates train/val splits after manual annotation is complete.

Usage:
    python create_splits.py --dataset datasets/modi_matra_sample --split 0.85
"""
import argparse
from pathlib import Path
import shutil
import random
from tqdm import tqdm
import json

def create_splits(dataset_dir, train_split=0.85):
    """Split annotated dataset into train/val."""
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / 'images'
    labels_dir = dataset_dir / 'labels'
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"ERROR: Dataset directory structure not found!")
        print(f"Expected: {images_dir} and {labels_dir}")
        return
    
    # Get all annotated images (those with non-empty label files)
    annotated = []
    empty_labels = []
    
    print("Scanning for annotated images...")
    for label_file in tqdm(list(labels_dir.glob('*.txt'))):
        # Skip classes.txt
        if label_file.name == 'classes.txt':
            continue
            
        if label_file.stat().st_size > 0:  # Non-empty
            # Find corresponding image
            img_file = None
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                test_file = images_dir / (label_file.stem + ext)
                if test_file.exists():
                    img_file = test_file
                    break
            
            if img_file and img_file.exists():
                annotated.append((img_file, label_file))
            else:
                print(f"Warning: No image found for label {label_file.name}")
        else:
            empty_labels.append(label_file.name)
    
    if len(annotated) == 0:
        print("\n" + "="*70)
        print("ERROR: No annotated images found!")
        print("="*70)
        print("\nPossible reasons:")
        print("1. You haven't started annotation yet")
        print("2. All label files are empty (no boxes drawn)")
        print("3. Label files are in wrong location")
        print(f"\nFound {len(empty_labels)} empty label files")
        if empty_labels:
            print(f"Examples: {empty_labels[:5]}")
        print("\nPlease annotate images using LabelImg first:")
        print(f"  labelImg {images_dir.absolute()} {labels_dir.absolute()}")
        return
    
    print(f"\n✓ Found {len(annotated)} annotated images")
    if empty_labels:
        print(f"⚠ Warning: {len(empty_labels)} label files are empty (not annotated)")
    
    # Load metadata if available
    metadata_file = dataset_dir / 'metadata.json'
    matra_counts = {}
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
            for item in metadata:
                img_name = item['image']
                matra_type = item['matra_type']
                # Check if this image was annotated
                if any(img_name == img.name for img, _ in annotated):
                    matra_counts[matra_type] = matra_counts.get(matra_type, 0) + 1
    
    if matra_counts:
        print("\nAnnotated by matra type:")
        for matra_type, count in matra_counts.items():
            print(f"  {matra_type}: {count}")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(annotated)
    
    split_idx = int(len(annotated) * train_split)
    train_data = annotated[:split_idx]
    val_data = annotated[split_idx:]
    
    print(f"\nSplit: Train={len(train_data)}, Val={len(val_data)}")
    
    # Create split directories
    train_img = images_dir / 'train'
    train_lbl = labels_dir / 'train'
    val_img = images_dir / 'val'
    val_lbl = labels_dir / 'val'
    
    for d in [train_img, train_lbl, val_img, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    print("\nCreating train split...")
    for img, lbl in tqdm(train_data):
        shutil.copy2(img, train_img / img.name)
        shutil.copy2(lbl, train_lbl / lbl.name)
    
    print("Creating val split...")
    for img, lbl in tqdm(val_data):
        shutil.copy2(img, val_img / img.name)
        shutil.copy2(lbl, val_lbl / lbl.name)
    
    # Create YAML config file for YOLO
    yaml_content = f"""# Modi Matra Detection Dataset
path: {dataset_dir.absolute()}
train: images/train
val: images/val

# Classes
nc: 3
names: ['top_matra', 'side_matra', 'bottom_matra']

# Dataset statistics
# Total annotated: {len(annotated)}
# Train: {len(train_data)} images
# Val: {len(val_data)} images
"""
    
    yaml_path = dataset_dir / 'modi_matra.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n{'='*70}")
    print(f"✓ Dataset splits created successfully!")
    print(f"{'='*70}")
    print(f"\nDataset: {dataset_dir.absolute()}")
    print(f"Train: {len(train_data)} images")
    print(f"Val: {len(val_data)} images")
    print(f"Config: {yaml_path}")
    
    print(f"\n📋 NEXT STEP - Train the model:")
    print(f"\nyolo detect train \\")
    print(f"    data={yaml_path} \\")
    print(f"    model=yolov8n.pt \\")
    print(f"    epochs=100 \\")
    print(f"    imgsz=640 \\")
    print(f"    batch=16 \\")
    print(f"    patience=20")
    print(f"\n{'='*70}")

def main():
    parser = argparse.ArgumentParser(
        description='Create train/val splits from annotated Modi matra dataset'
    )
    parser.add_argument('--dataset', required=True, help='Dataset directory (output from prepare_modi_dataset.py)')
    parser.add_argument('--split', type=float, default=0.85, help='Train split ratio (default: 0.85)')
    args = parser.parse_args()
    
    if not Path(args.dataset).exists():
        print(f"ERROR: Dataset not found: {args.dataset}")
        return
    
    if args.split <= 0 or args.split >= 1:
        print(f"ERROR: Split must be between 0 and 1 (got {args.split})")
        return
    
    create_splits(args.dataset, args.split)

if __name__ == '__main__':
    main()
