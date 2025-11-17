#!/usr/bin/env python3
"""
merge_datasets.py
Merge multiple annotated datasets into one combined training set.

Usage:
    python merge_datasets.py \
        --original datasets/modi_300_final \
        --high_conf datasets/modi_auto_labeled_6700/labels/high \
        --medium_reviewed datasets/modi_auto_labeled_6700/labels_reviewed/medium \
        --output datasets/modi_combined_5000
"""
import argparse
from pathlib import Path
import shutil
from tqdm import tqdm

def merge_datasets(*source_dirs, output_dir):
    """Merge multiple datasets into one."""
    
    output_dir = Path(output_dir)
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels').mkdir(parents=True, exist_ok=True)
    
    total_images = 0
    total_labels = 0
    
    print("Merging datasets...\n")
    
    for source_dir in source_dirs:
        source_dir = Path(source_dir)
        
        if not source_dir.exists():
            print(f"⚠️  Warning: {source_dir} not found, skipping...")
            continue
        
        print(f"Processing: {source_dir}")
        
        # Find images and labels
        images_dir = source_dir / 'images'
        labels_dir = source_dir / 'labels'
        
        # If source_dir itself contains images/labels directly
        if not images_dir.exists():
            images_dir = source_dir
            labels_dir = source_dir
        
        # Copy images
        image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg'))
        for img_file in tqdm(image_files, desc=f"  Copying from {source_dir.name}"):
            dst = output_dir / 'images' / img_file.name
            if not dst.exists():  # Avoid duplicates
                shutil.copy(img_file, dst)
                total_images += 1
                
                # Copy corresponding label
                label_file = labels_dir / (img_file.stem + '.txt')
                if label_file.exists():
                    dst_label = output_dir / 'labels' / label_file.name
                    shutil.copy(label_file, dst_label)
                    total_labels += 1
    
    # Check class distribution
    class_counts = {0: 0, 1: 0, 2: 0}
    for label_file in (output_dir / 'labels').glob('*.txt'):
        if label_file.stat().st_size > 0:
            with open(label_file) as f:
                for line in f:
                    cls = int(line.strip().split()[0])
                    class_counts[cls] += 1
    
    print(f"\n{'='*70}")
    print("MERGE COMPLETE!")
    print("="*70)
    print(f"Total images: {total_images}")
    print(f"Total labels: {total_labels}")
    print(f"\nClass distribution:")
    print(f"  top_matra (0):    {class_counts[0]}")
    print(f"  side_matra (1):   {class_counts[1]}")
    print(f"  bottom_matra (2): {class_counts[2]}")
    print(f"\nOutput: {output_dir}")
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description='Merge multiple datasets')
    parser.add_argument('--original', help='Original training dataset')
    parser.add_argument('--high_conf', help='High confidence auto-labeled')
    parser.add_argument('--medium_reviewed', help='Reviewed medium confidence')
    parser.add_argument('--low_annotated', help='Manually annotated low confidence')
    parser.add_argument('--output', required=True, help='Output merged dataset')
    
    args = parser.parse_args()
    
    # Collect all source directories
    sources = []
    for source in [args.original, args.high_conf, args.medium_reviewed, args.low_annotated]:
        if source:
            sources.append(source)
    
    if len(sources) == 0:
        print("Error: No source datasets provided!")
        return
    
    merge_datasets(*sources, output_dir=args.output)

if __name__ == '__main__':
    main()
