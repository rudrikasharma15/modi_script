#!/usr/bin/env python3
"""
Create exclusion list from labeled dataset.
Extracts unique image filenames from train/val splits.
"""
import argparse
from pathlib import Path

def create_exclusion_list(dataset_dir, output_file):
    """Extract unique image filenames from train/val directories."""
    
    dataset_path = Path(dataset_dir)
    unique_images = set()
    
    # Check train and val directories
    for split in ['train', 'val']:
        # Check images directory
        img_dir = dataset_path / 'images' / split
        if img_dir.exists():
            for img in img_dir.iterdir():
                if img.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    unique_images.add(img.name)
                    print(f"Found image: {img.name}")
        
        # Also check labels directory to find corresponding images
        label_dir = dataset_path / 'labels' / split
        if label_dir.exists():
            for label in label_dir.iterdir():
                if label.suffix == '.txt' and label.name != 'classes.txt':
                    # Find corresponding image
                    stem = label.stem
                    # Check for actual image file
                    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                        img_path = img_dir / f"{stem}{ext}"
                        if img_path.exists():
                            unique_images.add(img_path.name)
                            break
    
    # Write to file
    with open(output_file, 'w') as f:
        for img_name in sorted(unique_images):
            f.write(f"{img_name}\n")
    
    print(f"\n✓ Created exclusion list: {output_file}")
    print(f"✓ Total unique images to exclude: {len(unique_images)}")
    
    return len(unique_images)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create exclusion list from labeled dataset')
    parser.add_argument('--dataset', required=True, help='Path to labeled dataset directory')
    parser.add_argument('--output', default='already_labeled.txt', help='Output file name')
    
    args = parser.parse_args()
    
    count = create_exclusion_list(args.dataset, args.output)
    
    print(f"\n📝 Use this file with:")
    print(f"python auto_label_full_dataset.py --exclusion_file {args.output} ...")