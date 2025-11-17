# #!/usr/bin/env python3
# """
# create_splits.py
# Creates train/val splits after manual annotation is complete.

# Usage:
#     python create_splits.py --dataset datasets/modi_matra_sample --split 0.85
# """
# import argparse
# from pathlib import Path
# import shutil
# import random
# from tqdm import tqdm
# import json

# def create_splits(dataset_dir, train_split=0.85):
#     """Split annotated dataset into train/val."""
#     dataset_dir = Path(dataset_dir)
#     images_dir = dataset_dir / 'images'
#     labels_dir = dataset_dir / 'labels'
    
#     if not images_dir.exists() or not labels_dir.exists():
#         print(f"ERROR: Dataset directory structure not found!")
#         print(f"Expected: {images_dir} and {labels_dir}")
#         return
    
#     # Get all annotated images (those with non-empty label files)
#     annotated = []
#     empty_labels = []
    
#     print("Scanning for annotated images...")
#     for label_file in tqdm(list(labels_dir.glob('*.txt'))):
#         # Skip classes.txt
#         if label_file.name == 'classes.txt':
#             continue
            
#         if label_file.stat().st_size > 0:  # Non-empty
#             # Find corresponding image
#             img_file = None
#             for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
#                 test_file = images_dir / (label_file.stem + ext)
#                 if test_file.exists():
#                     img_file = test_file
#                     break
            
#             if img_file and img_file.exists():
#                 annotated.append((img_file, label_file))
#             else:
#                 print(f"Warning: No image found for label {label_file.name}")
#         else:
#             empty_labels.append(label_file.name)
    
#     if len(annotated) == 0:
#         print("\n" + "="*70)
#         print("ERROR: No annotated images found!")
#         print("="*70)
#         print("\nPossible reasons:")
#         print("1. You haven't started annotation yet")
#         print("2. All label files are empty (no boxes drawn)")
#         print("3. Label files are in wrong location")
#         print(f"\nFound {len(empty_labels)} empty label files")
#         if empty_labels:
#             print(f"Examples: {empty_labels[:5]}")
#         print("\nPlease annotate images using LabelImg first:")
#         print(f"  labelImg {images_dir.absolute()} {labels_dir.absolute()}")
#         return
    
#     print(f"\n✓ Found {len(annotated)} annotated images")
#     if empty_labels:
#         print(f"⚠ Warning: {len(empty_labels)} label files are empty (not annotated)")
    
#     # Load metadata if available
#     metadata_file = dataset_dir / 'metadata.json'
#     matra_counts = {}
#     if metadata_file.exists():
#         with open(metadata_file) as f:
#             metadata = json.load(f)
#             for item in metadata:
#                 img_name = item['image']
#                 matra_type = item['matra_type']
#                 # Check if this image was annotated
#                 if any(img_name == img.name for img, _ in annotated):
#                     matra_counts[matra_type] = matra_counts.get(matra_type, 0) + 1
    
#     if matra_counts:
#         print("\nAnnotated by matra type:")
#         for matra_type, count in matra_counts.items():
#             print(f"  {matra_type}: {count}")
    
#     # Shuffle and split
#     random.seed(42)
#     random.shuffle(annotated)
    
#     split_idx = int(len(annotated) * train_split)
#     train_data = annotated[:split_idx]
#     val_data = annotated[split_idx:]
    
#     print(f"\nSplit: Train={len(train_data)}, Val={len(val_data)}")
    
#     # Create split directories
#     train_img = images_dir / 'train'
#     train_lbl = labels_dir / 'train'
#     val_img = images_dir / 'val'
#     val_lbl = labels_dir / 'val'
    
#     for d in [train_img, train_lbl, val_img, val_lbl]:
#         d.mkdir(parents=True, exist_ok=True)
    
#     # Copy files
#     print("\nCreating train split...")
#     for img, lbl in tqdm(train_data):
#         shutil.copy2(img, train_img / img.name)
#         shutil.copy2(lbl, train_lbl / lbl.name)
    
#     print("Creating val split...")
#     for img, lbl in tqdm(val_data):
#         shutil.copy2(img, val_img / img.name)
#         shutil.copy2(lbl, val_lbl / lbl.name)
    
#     # Create YAML config file for YOLO
#     yaml_content = f"""# Modi Matra Detection Dataset
# path: {dataset_dir.absolute()}
# train: images/train
# val: images/val

# # Classes
# nc: 3
# names: ['top_matra', 'side_matra', 'bottom_matra']

# # Dataset statistics
# # Total annotated: {len(annotated)}
# # Train: {len(train_data)} images
# # Val: {len(val_data)} images
# """
    
#     yaml_path = dataset_dir / 'modi_matra.yaml'
#     with open(yaml_path, 'w') as f:
#         f.write(yaml_content)
    
#     print(f"\n{'='*70}")
#     print(f"✓ Dataset splits created successfully!")
#     print(f"{'='*70}")
#     print(f"\nDataset: {dataset_dir.absolute()}")
#     print(f"Train: {len(train_data)} images")
#     print(f"Val: {len(val_data)} images")
#     print(f"Config: {yaml_path}")
    
#     print(f"\n📋 NEXT STEP - Train the model:")
#     print(f"\nyolo detect train \\")
#     print(f"    data={yaml_path} \\")
#     print(f"    model=yolov8n.pt \\")
#     print(f"    epochs=100 \\")
#     print(f"    imgsz=640 \\")
#     print(f"    batch=16 \\")
#     print(f"    patience=20")
#     print(f"\n{'='*70}")

# def main():
#     parser = argparse.ArgumentParser(
#         description='Create train/val splits from annotated Modi matra dataset'
#     )
#     parser.add_argument('--dataset', required=True, help='Dataset directory (output from prepare_modi_dataset.py)')
#     parser.add_argument('--split', type=float, default=0.85, help='Train split ratio (default: 0.85)')
#     args = parser.parse_args()
    
#     if not Path(args.dataset).exists():
#         print(f"ERROR: Dataset not found: {args.dataset}")
#         return
    
#     if args.split <= 0 or args.split >= 1:
#         print(f"ERROR: Split must be between 0 and 1 (got {args.split})")
#         return
    
#     create_splits(args.dataset, args.split)

# if __name__ == '__main__':
#     main()
#!/usr/bin/env python3
"""
create_splits.py - Create train/val/test splits with proper distribution
"""
import argparse
from pathlib import Path
import shutil
import random
from collections import defaultdict

def create_splits(dataset_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Create train/val/test splits ensuring balanced class distribution.
    
    Args:
        dataset_dir: Dataset root with images/ and labels/
        train_ratio: Training split ratio (default 0.70)
        val_ratio: Validation split ratio (default 0.15)
        test_ratio: Test split ratio (default 0.15)
    """
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / 'images'
    labels_dir = dataset_dir / 'labels'
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"ERROR: {images_dir} or {labels_dir} not found!")
        return
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        print(f"ERROR: Ratios must sum to 1.0 (got {train_ratio + val_ratio + test_ratio})")
        return
    
    # Find all annotated images (have non-empty label files)
    label_files = [f for f in labels_dir.glob('*.txt') 
                   if f.stat().st_size > 0 and f.name != 'classes.txt']
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    annotated_pairs = []
    
    for label_file in label_files:
        # Find corresponding image
        img_name = label_file.stem
        img_file = None
        for ext in image_extensions:
            candidate = images_dir / (img_name + ext)
            if candidate.exists():
                img_file = candidate
                break
        
        if img_file:
            annotated_pairs.append((img_file, label_file))
    
    if not annotated_pairs:
        print(f"ERROR: No annotated images found!")
        return
    
    print(f"\n✓ Found {len(annotated_pairs)} annotated images")
    
    # Group by class for stratified splitting
    images_by_class = defaultdict(list)
    
    for img_file, label_file in annotated_pairs:
        # Read label file to get classes
        with open(label_file) as f:
            classes = set()
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes.add(int(parts[0]))
        
        # Use primary class (first one, or most common)
        if classes:
            primary_class = sorted(classes)[0]  # Use lowest class ID as primary
            images_by_class[primary_class].append((img_file, label_file))
    
    # Print class distribution
    class_names = {0: 'top_matra', 1: 'side_matra', 2: 'bottom_matra'}
    print("\n📊 Class distribution:")
    for class_id in sorted(images_by_class.keys()):
        count = len(images_by_class[class_id])
        class_name = class_names.get(class_id, f'class_{class_id}')
        print(f"  {class_name}: {count} images")
    
    # Stratified split by class
    random.seed(42)
    
    train_pairs = []
    val_pairs = []
    test_pairs = []
    
    for class_id, pairs in images_by_class.items():
        random.shuffle(pairs)
        
        n = len(pairs)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_pairs.extend(pairs[:train_end])
        val_pairs.extend(pairs[train_end:val_end])
        test_pairs.extend(pairs[val_end:])
    
    # Shuffle within splits
    random.shuffle(train_pairs)
    random.shuffle(val_pairs)
    random.shuffle(test_pairs)
    
    print(f"\n📂 Split distribution:")
    print(f"  Training:   {len(train_pairs)} images ({len(train_pairs)/len(annotated_pairs)*100:.1f}%)")
    print(f"  Validation: {len(val_pairs)} images ({len(val_pairs)/len(annotated_pairs)*100:.1f}%)")
    print(f"  Test:       {len(test_pairs)} images ({len(test_pairs)/len(annotated_pairs)*100:.1f}%)")
    
    # Check if test set is reasonable size
    if len(test_pairs) < 20:
        print(f"\n⚠️  WARNING: Test set only has {len(test_pairs)} images!")
        print(f"    Recommendation: Use at least 50 images for reliable evaluation")
        print(f"    Consider collecting more data or adjusting split ratios")
    
    # Create split directories
    for split in ['train', 'val', 'test']:
        (images_dir / split).mkdir(exist_ok=True)
        (labels_dir / split).mkdir(exist_ok=True)
    
    # Copy files to respective splits
    def copy_split(pairs, split_name):
        for img_file, label_file in pairs:
            shutil.copy2(img_file, images_dir / split_name / img_file.name)
            shutil.copy2(label_file, labels_dir / split_name / label_file.name)
    
    print(f"\n📋 Creating splits...")
    copy_split(train_pairs, 'train')
    copy_split(val_pairs, 'val')
    copy_split(test_pairs, 'test')
    
    # Create YOLO data config
    yaml_content = f"""# Modi Matra Detection Dataset Configuration

# Dataset paths
path: {dataset_dir.absolute()}
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: top_matra
  1: side_matra
  2: bottom_matra

# Class descriptions
descriptions:
  top_matra: "Matra marks appearing above the base character (i, e, ai, anusvara)"
  side_matra: "Matra marks appearing on the right side (aa, o, au, visarga)"
  bottom_matra: "Matra marks appearing below the base character (u, uu)"

# Dataset statistics
total_images: {len(annotated_pairs)}
train_images: {len(train_pairs)}
val_images: {len(val_pairs)}
test_images: {len(test_pairs)}

# Split ratios
train_ratio: {train_ratio}
val_ratio: {val_ratio}
test_ratio: {test_ratio}
"""
    
    yaml_path = dataset_dir / 'modi_matra.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ Dataset splits created successfully!")
    print(f"\n📄 YOLO config saved to: {yaml_path}")
    
    # Print per-class distribution in each split
    print(f"\n📊 Per-class distribution in splits:")
    
    def count_classes_in_split(pairs):
        counts = defaultdict(int)
        for _, label_file in pairs:
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        counts[int(parts[0])] += 1
        return counts
    
    train_counts = count_classes_in_split(train_pairs)
    val_counts = count_classes_in_split(val_pairs)
    test_counts = count_classes_in_split(test_pairs)
    
    print(f"\n  {'Class':<15} {'Train':<10} {'Val':<10} {'Test':<10}")
    print(f"  {'-'*45}")
    for class_id in sorted(set(train_counts.keys()) | set(val_counts.keys()) | set(test_counts.keys())):
        class_name = class_names.get(class_id, f'class_{class_id}')
        print(f"  {class_name:<15} {train_counts[class_id]:<10} {val_counts[class_id]:<10} {test_counts[class_id]:<10}")
    
    print(f"\n{'='*70}")
    print(f"Next steps:")
    print(f"{'='*70}")
    print(f"\n1. Train model:")
    print(f"   python train_modi_matra.py \\")
    print(f"       --data {yaml_path} \\")
    print(f"       --epochs 150 \\")
    print(f"       --batch 16")
    
    print(f"\n2. Validate during training:")
    print(f"   Uses {len(val_pairs)} validation images automatically")
    
    print(f"\n3. Final evaluation on held-out test set:")
    print(f"   python predict_modi_matra.py \\")
    print(f"       --model runs/modi_matra/train/weights/best.pt \\")
    print(f"       --source {images_dir / 'test'} \\")
    print(f"       --output test_predictions/")
    
    print(f"\n4. Report test set results in your paper:")
    print(f"   - Test set size: {len(test_pairs)} images")
    print(f"   - Never used during training or validation")
    print(f"   - Unbiased final performance estimate")
    
    print(f"\n{'='*70}")
    
    return len(train_pairs), len(val_pairs), len(test_pairs)

def main():
    parser = argparse.ArgumentParser(
        description='Create train/val/test splits for Modi matra dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard 70/15/15 split (recommended for papers)
  python create_splits.py --dataset datasets/modi_300
  
  # Custom split ratios
  python create_splits.py --dataset datasets/modi_300 --train 0.70 --val 0.20 --test 0.10
  
  # For small datasets (80/10/10)
  python create_splits.py --dataset datasets/modi_100 --train 0.80 --val 0.10 --test 0.10
        """
    )
    parser.add_argument('--dataset', required=True, help='Dataset directory')
    parser.add_argument('--train', type=float, default=0.70, help='Training split ratio (default: 0.70)')
    parser.add_argument('--val', type=float, default=0.15, help='Validation split ratio (default: 0.15)')
    parser.add_argument('--test', type=float, default=0.15, help='Test split ratio (default: 0.15)')
    args = parser.parse_args()
    
    create_splits(args.dataset, args.train, args.val, args.test)

if __name__ == '__main__':
    main()