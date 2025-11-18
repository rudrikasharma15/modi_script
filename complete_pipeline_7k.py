#!/usr/bin/env python3
"""
Complete pipeline to auto-label remaining images and create proper train/val/test splits.
Strategy: 60% train, 20% val, 20% test with stratified sampling
"""

import argparse
import shutil
import random
import yaml
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
import cv2
from tqdm import tqdm

# ============================================================================
# STEP 1: Auto-label remaining 2K images
# ============================================================================

def auto_label_remaining_images(model_path, data_root, already_used_dir, output_dir, conf_threshold=0.5):
    """
    Auto-label images that weren't in the already_used dataset.
    """
    print("\n" + "="*70)
    print("STEP 1: AUTO-LABELING REMAINING IMAGES")
    print("="*70)
    
    model = YOLO(model_path)
    data_root = Path(data_root)
    already_used_dir = Path(already_used_dir)
    output_dir = Path(output_dir)
    
    # Get already used images
    used_images = set()
    if already_used_dir.exists():
        for img_path in already_used_dir.rglob("*.jpg"):
            used_images.add(img_path.name)
        for img_path in already_used_dir.rglob("*.png"):
            used_images.add(img_path.name)
    
    print(f"📊 Already used images: {len(used_images)}")
    
    # Find all images in data_root
    all_images = []
    for img_path in data_root.rglob("*.jpg"):
        if img_path.name not in used_images:
            all_images.append(img_path)
    for img_path in data_root.rglob("*.png"):
        if img_path.name not in used_images:
            all_images.append(img_path)
    
    print(f"📊 New images to label: {len(all_images)}")
    
    if len(all_images) == 0:
        print("⚠️  No new images found!")
        return output_dir
    
    # Create output directories
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    # Statistics
    stats = {
        'total': 0,
        'with_detections': 0,
        'high_conf': 0,
        'medium_conf': 0,
        'low_conf': 0
    }
    
    print(f"\n🔍 Running predictions on {len(all_images)} images...")
    
    for img_path in tqdm(all_images, desc="Auto-labeling"):
        try:
            # Run prediction
            results = model.predict(img_path, conf=0.25, verbose=False)[0]
            
            stats['total'] += 1
            
            if len(results.boxes) == 0:
                continue
            
            stats['with_detections'] += 1
            
            # Copy image
            shutil.copy(img_path, output_dir / "images" / img_path.name)
            
            # Save YOLO format labels
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            
            label_path = output_dir / "labels" / f"{img_path.stem}.txt"
            
            max_conf = 0
            with open(label_path, 'w') as f:
                for box in results.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0]
                    
                    # Convert to YOLO format (normalized center_x, center_y, width, height)
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    
                    f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
                    max_conf = max(max_conf, conf)
            
            # Track confidence
            if max_conf >= 0.8:
                stats['high_conf'] += 1
            elif max_conf >= 0.5:
                stats['medium_conf'] += 1
            else:
                stats['low_conf'] += 1
                
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue
    
    print("\n" + "="*70)
    print("AUTO-LABELING SUMMARY")
    print("="*70)
    print(f"Total images processed: {stats['total']}")
    print(f"With detections: {stats['with_detections']}")
    print(f"  High confidence (>0.8): {stats['high_conf']}")
    print(f"  Medium confidence (0.5-0.8): {stats['medium_conf']}")
    print(f"  Low confidence (<0.5): {stats['low_conf']}")
    print(f"\n✅ Saved to: {output_dir}")
    print("="*70)
    
    return output_dir

# ============================================================================
# STEP 2: Merge all datasets
# ============================================================================

def merge_all_datasets(manual_dir, auto_labeled_dir, output_dir):
    """
    Merge manual annotations with auto-labeled images.
    """
    print("\n" + "="*70)
    print("STEP 2: MERGING ALL DATASETS")
    print("="*70)
    
    manual_dir = Path(manual_dir)
    auto_labeled_dir = Path(auto_labeled_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    total_images = 0
    
    # Copy manual annotations
    print("\n📂 Copying manual annotations...")
    if manual_dir.exists():
        manual_images = list((manual_dir / "images").glob("*"))
        for img_path in tqdm(manual_images, desc="Manual images"):
            shutil.copy(img_path, output_dir / "images" / img_path.name)
            
            label_path = manual_dir / "labels" / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy(label_path, output_dir / "labels" / f"{img_path.stem}.txt")
        
        total_images += len(manual_images)
        print(f"  ✅ Copied {len(manual_images)} manual images")
    
    # Copy auto-labeled
    print("\n📂 Copying auto-labeled images...")
    if auto_labeled_dir.exists():
        auto_images = list((auto_labeled_dir / "images").glob("*"))
        for img_path in tqdm(auto_images, desc="Auto-labeled images"):
            if not (output_dir / "images" / img_path.name).exists():
                shutil.copy(img_path, output_dir / "images" / img_path.name)
                
                label_path = auto_labeled_dir / "labels" / f"{img_path.stem}.txt"
                if label_path.exists():
                    shutil.copy(label_path, output_dir / "labels" / f"{img_path.stem}.txt")
        
        auto_count = len(list((output_dir / "images").glob("*"))) - total_images
        total_images = len(list((output_dir / "images").glob("*")))
        print(f"  ✅ Added {auto_count} auto-labeled images")
    
    print(f"\n📊 Total merged images: {total_images}")
    print(f"✅ Saved to: {output_dir}")
    print("="*70)
    
    return output_dir

# ============================================================================
# STEP 3: Create stratified train/val/test splits (60/20/20)
# ============================================================================

def create_stratified_splits(dataset_dir, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, seed=42):
    """
    Create stratified train/val/test splits ensuring class balance.
    """
    print("\n" + "="*70)
    print("STEP 3: CREATING STRATIFIED SPLITS (60/20/20)")
    print("="*70)
    
    dataset_dir = Path(dataset_dir)
    random.seed(seed)
    
    # Get all labeled images
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    
    # Group images by class distribution
    print("\n📊 Analyzing class distribution...")
    
    image_classes = defaultdict(set)  # class -> set of images
    image_to_classes = {}  # image -> list of classes
    
    for label_path in labels_dir.glob("*.txt"):
        img_name = label_path.stem
        
        # Find corresponding image
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            candidate = images_dir / f"{img_name}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        
        if not img_path:
            continue
        
        # Read classes in this image
        classes_in_image = set()
        with open(label_path, 'r') as f:
            for line in f:
                cls = int(line.split()[0])
                classes_in_image.add(cls)
        
        image_to_classes[img_name] = classes_in_image
        
        # Add to each class's image set
        for cls in classes_in_image:
            image_classes[cls].add(img_name)
    
    # Print class distribution
    class_names = {0: 'top_matra', 1: 'side_matra', 2: 'bottom_matra'}
    print("\nClass distribution:")
    for cls in sorted(image_classes.keys()):
        print(f"  {class_names.get(cls, f'class_{cls}')}: {len(image_classes[cls])} images")
    
    # Stratified split
    print("\n✂️  Creating stratified splits...")
    
    train_set = set()
    val_set = set()
    test_set = set()
    
    # For each class, split proportionally
    for cls, img_set in image_classes.items():
        img_list = list(img_set)
        random.shuffle(img_list)
        
        n_total = len(img_list)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_set.update(img_list[:n_train])
        val_set.update(img_list[n_train:n_train + n_val])
        test_set.update(img_list[n_train + n_val:])
    
    # Resolve overlaps (images with multiple classes)
    # Priority: train > val > test (keep training set largest)
    val_set -= train_set
    test_set -= train_set
    test_set -= val_set
    
    print(f"\n📊 Split sizes:")
    print(f"  Train: {len(train_set)} images ({len(train_set)/len(image_to_classes)*100:.1f}%)")
    print(f"  Val:   {len(val_set)} images ({len(val_set)/len(image_to_classes)*100:.1f}%)")
    print(f"  Test:  {len(test_set)} images ({len(test_set)/len(image_to_classes)*100:.1f}%)")
    
    # Create split directories
    for split in ['train', 'val', 'test']:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Move files to split directories
    print("\n📦 Organizing files into splits...")
    
    for img_name, classes in tqdm(image_to_classes.items(), desc="Moving files"):
        # Find image file
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            candidate = images_dir / f"{img_name}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        
        if not img_path:
            continue
        
        # Determine split
        if img_name in train_set:
            split = 'train'
        elif img_name in val_set:
            split = 'val'
        elif img_name in test_set:
            split = 'test'
        else:
            continue
        
        # Move image and label
        shutil.move(str(img_path), str(dataset_dir / "images" / split / img_path.name))
        
        label_path = labels_dir / f"{img_name}.txt"
        if label_path.exists():
            shutil.move(str(label_path), str(dataset_dir / "labels" / split / f"{img_name}.txt"))
    
    # Verify class distribution in each split
    print("\n📊 Verifying class distribution in splits:")
    
    for split in ['train', 'val', 'test']:
        split_labels = dataset_dir / "labels" / split
        class_counts = defaultdict(int)
        
        for label_path in split_labels.glob("*.txt"):
            with open(label_path, 'r') as f:
                for line in f:
                    cls = int(line.split()[0])
                    class_counts[cls] += 1
        
        print(f"\n  {split.upper()}:")
        for cls in sorted(class_counts.keys()):
            print(f"    {class_names.get(cls, f'class_{cls}')}: {class_counts[cls]} instances")
    
    # Create YAML file
    yaml_content = {
        'path': str(dataset_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {
            0: 'top_matra',
            1: 'side_matra',
            2: 'bottom_matra'
        }
    }
    
    yaml_path = dataset_dir / 'modi_matra.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"\n✅ YAML file created: {yaml_path}")
    print("="*70)
    
    return dataset_dir

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Complete pipeline: auto-label + merge + split')
    parser.add_argument('--model', required=True, help='Path to trained model for auto-labeling')
    parser.add_argument('--data_root', required=True, help='Root directory with all 7K images')
    parser.add_argument('--already_used', required=True, help='Directory with already annotated images')
    parser.add_argument('--output', default='datasets/modi_full_7k', help='Output directory')
    parser.add_argument('--conf_threshold', type=float, default=0.5, help='Confidence threshold for auto-labeling')
    parser.add_argument('--train_ratio', type=float, default=0.60, help='Train split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.20, help='Validation split ratio')
    parser.add_argument('--test_ratio', type=float, default=0.20, help='Test split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.01:
        print("❌ Error: train_ratio + val_ratio + test_ratio must equal 1.0")
        return
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("🎯 MODI MATRA COMPLETE PIPELINE")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Data root: {args.data_root}")
    print(f"Already used: {args.already_used}")
    print(f"Output: {args.output}")
    print(f"Split ratio: {args.train_ratio:.0%}/{args.val_ratio:.0%}/{args.test_ratio:.0%}")
    print("="*70)
    
    # Step 1: Auto-label remaining images
    auto_labeled_dir = output_dir / "auto_labeled"
    auto_label_remaining_images(
        args.model,
        args.data_root,
        args.already_used,
        auto_labeled_dir,
        args.conf_threshold
    )
    
    # Step 2: Merge all datasets
    merged_dir = output_dir / "merged"
    merge_all_datasets(
        args.already_used,
        auto_labeled_dir,
        merged_dir
    )
    
    # Step 3: Create stratified splits
    final_dataset = create_stratified_splits(
        merged_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    print(f"\n📁 Final dataset: {final_dataset}")
    print(f"📄 YAML config: {final_dataset / 'modi_matra.yaml'}")
    print("\nNext steps:")
    print("="*70)
    print("1. Train on full dataset:")
    print(f"   python train_modi_matra.py --data {final_dataset / 'modi_matra.yaml'} --epochs 150 --batch 32 --name train_7k")
    print("\n2. Evaluate on test set:")
    print(f"   yolo detect val model=runs/modi_matra/train_7k/weights/best.pt data={final_dataset / 'modi_matra.yaml'} split=test")
    print("="*70)

if __name__ == '__main__':
    main()