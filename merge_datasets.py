#!/usr/bin/env python3
"""
Merge multiple datasets into a single combined dataset.
Handles duplicates by keeping only one copy of each unique image.
"""
import argparse
import shutil
from pathlib import Path
from collections import Counter
from tqdm import tqdm

def find_image_for_label(label_file, search_dirs):
    """Find corresponding image file for a label."""
    stem = label_file.stem
    
    for search_dir in search_dirs:
        search_path = Path(search_dir)
        if not search_path.exists():
            continue
            
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            img_path = search_path / f"{stem}{ext}"
            if img_path.exists():
                return img_path
    
    return None

def merge_datasets(original_dir, high_conf_labels, medium_reviewed_labels, output_dir):
    """Merge original, high confidence, and reviewed medium confidence datasets."""
    
    output_dir = Path(output_dir)
    
    # Create output directories
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels').mkdir(parents=True, exist_ok=True)
    
    print("🔄 Merging datasets...\n")
    
    # Track all processed images to avoid duplicates
    processed_stems = set()
    stats = Counter()
    
    # Priority 1: Original manually-labeled dataset (highest quality)
    if original_dir:
        original_path = Path(original_dir)
        print(f"Processing: {original_path}")
        
        # Collect all images from train/val
        all_imgs = []
        for split in ['train', 'val']:
            img_dir = original_path / 'images' / split
            if img_dir.exists():
                all_imgs.extend(list(img_dir.glob('*.png')))
                all_imgs.extend(list(img_dir.glob('*.jpg')))
                all_imgs.extend(list(img_dir.glob('*.jpeg')))
        
        for img_file in tqdm(all_imgs, desc=f"  Copying from {original_path.name}"):
            img_stem = img_file.stem
            
            if img_stem not in processed_stems:
                # Copy image
                dst_img = output_dir / 'images' / img_file.name
                shutil.copy2(img_file, dst_img)
                
                # Find and copy label
                for split in ['train', 'val']:
                    label_file = original_path / 'labels' / split / (img_stem + '.txt')
                    if label_file.exists():
                        dst_label = output_dir / 'labels' / (img_stem + '.txt')
                        shutil.copy2(label_file, dst_label)
                        stats['original'] += 1
                        processed_stems.add(img_stem)
                        break
        
        print(f"  ✓ Added {stats['original']} original images\n")
    
    # Priority 2: High confidence auto-labeled
    if high_conf_labels:
        high_conf_path = Path(high_conf_labels)
        print(f"Processing: {high_conf_path}")
        
        # Find label files
        label_files = list(high_conf_path.glob('*.txt'))
        
        # Possible image locations
        img_search_dirs = [
            high_conf_path.parent.parent / 'images' / 'high',
            high_conf_path.parent / 'images' / 'high',
            high_conf_path / '../images/high',
        ]
        
        if len(label_files) == 0:
            print(f"  ⚠️  No label files found in {high_conf_path}")
        else:
            for label_file in tqdm(label_files, desc=f"  Copying from high"):
                img_stem = label_file.stem
                
                if img_stem not in processed_stems:
                    # Find corresponding image
                    img_file = find_image_for_label(label_file, img_search_dirs)
                    
                    if img_file:
                        # Copy image
                        dst_img = output_dir / 'images' / img_file.name
                        shutil.copy2(img_file, dst_img)
                        
                        # Copy label
                        dst_label = output_dir / 'labels' / (img_stem + '.txt')
                        shutil.copy2(label_file, dst_label)
                        stats['high_conf'] += 1
                        processed_stems.add(img_stem)
                    else:
                        stats['high_missing_img'] += 1
                else:
                    stats['duplicates_skipped'] += 1
        
        print(f"  ✓ Added {stats['high_conf']} high confidence images")
        if stats['duplicates_skipped'] > 0:
            print(f"  ⊘ Skipped {stats['duplicates_skipped']} duplicates")
        if stats['high_missing_img'] > 0:
            print(f"  ⚠️  Missing images: {stats['high_missing_img']}")
        print()
    
    # Priority 3: Reviewed medium confidence
    if medium_reviewed_labels:
        medium_path = Path(medium_reviewed_labels)
        print(f"Processing: {medium_path}")
        
        # Find label files
        label_files = list(medium_path.glob('*.txt'))
        
        # Possible image locations
        img_search_dirs = [
            medium_path.parent.parent / 'images' / 'medium',
            medium_path.parent / 'images' / 'medium',
            medium_path.parent.parent / 'labels_json' / 'medium',  # Images might be with JSONs
        ]
        
        if len(label_files) == 0:
            print(f"  ⚠️  No label files found in {medium_path}")
        else:
            dup_count = 0
            missing_count = 0
            
            for label_file in tqdm(label_files, desc=f"  Copying from medium_reviewed"):
                img_stem = label_file.stem
                
                if img_stem not in processed_stems:
                    # Find corresponding image
                    img_file = find_image_for_label(label_file, img_search_dirs)
                    
                    if img_file:
                        # Copy image
                        dst_img = output_dir / 'images' / img_file.name
                        shutil.copy2(img_file, dst_img)
                        
                        # Copy label
                        dst_label = output_dir / 'labels' / (img_stem + '.txt')
                        shutil.copy2(label_file, dst_label)
                        stats['medium_reviewed'] += 1
                        processed_stems.add(img_stem)
                    else:
                        missing_count += 1
                else:
                    dup_count += 1
        
        print(f"  ✓ Added {stats['medium_reviewed']} medium reviewed images")
        if dup_count > 0:
            print(f"  ⊘ Skipped {dup_count} duplicates")
        if missing_count > 0:
            print(f"  ⚠️  Missing images: {missing_count}")
        print()
    
    # Calculate class distribution
    print("📊 Analyzing class distribution...")
    class_counts = Counter()
    
    for label_file in (output_dir / 'labels').iterdir():
        if label_file.suffix == '.txt':
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            class_id = int(line.split()[0])
                            class_counts[class_id] += 1
                        except:
                            pass
    
    # Summary
    total_images = len(list((output_dir / 'images').iterdir()))
    total_labels = len(list((output_dir / 'labels').iterdir()))
    
    print("\n" + "="*70)
    print("🎉 MERGE COMPLETE!")
    print("="*70)
    print(f"\n📊 Dataset Summary:")
    print(f"  Total images: {total_images:,}")
    print(f"  Total labels: {total_labels:,}")
    print(f"\n📈 Sources:")
    print(f"  Original (manual):     {stats['original']:,}")
    print(f"  High confidence:       {stats['high_conf']:,}")
    print(f"  Medium reviewed:       {stats['medium_reviewed']:,}")
    if stats['duplicates_skipped'] > 0:
        print(f"  Duplicates skipped:    {stats['duplicates_skipped']:,}")
    
    print(f"\n🏷️  Class Distribution:")
    class_names = {0: "top_matra", 1: "side_matra", 2: "bottom_matra"}
    for class_id in sorted(class_counts.keys()):
        class_name = class_names.get(class_id, f"class_{class_id}")
        print(f"  {class_name} ({class_id}): {class_counts[class_id]:,}")
    
    print(f"\n💾 Output directory: {output_dir}")
    print("="*70 + "\n")
    
    print("📝 Next steps:")
    print("  1. Create train/val/test splits:")
    print(f"     python create_splits.py --dataset {output_dir}")
    print("  2. Train model:")
    print(f"     python train_modi_matra.py --data {output_dir}/modi_matra.yaml --epochs 200")
    
    return total_images

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge multiple datasets')
    parser.add_argument('--original', help='Original manually-labeled dataset directory')
    parser.add_argument('--high_conf', help='High confidence labels directory')
    parser.add_argument('--medium_reviewed', help='Reviewed medium confidence labels directory')
    parser.add_argument('--output', required=True, help='Output directory for merged dataset')
    
    args = parser.parse_args()
    
    merge_datasets(args.original, args.high_conf, args.medium_reviewed, args.output)