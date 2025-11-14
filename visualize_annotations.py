#!/usr/bin/env python3
"""
visualize_annotations.py
Visualizes annotated matra bounding boxes on images.

Usage:
    # Visualize all annotated images
    python visualize_annotations.py --dataset datasets/modi_matra_sample
    
    # Visualize specific split
    python visualize_annotations.py --dataset datasets/modi_matra_sample --split train
    
    # Save visualizations
    python visualize_annotations.py --dataset datasets/modi_matra_sample --output viz_output
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import random

# Class colors (BGR format for OpenCV)
CLASS_COLORS = {
    0: (0, 255, 0),      # top_matra - Green
    1: (255, 0, 0),      # side_matra - Blue
    2: (0, 0, 255),      # bottom_matra - Red
}

CLASS_NAMES = {
    0: 'top_matra',
    1: 'side_matra',
    2: 'bottom_matra'
}

def draw_yolo_box(image, class_id, x_center, y_center, width, height):
    """Draw YOLO format bounding box on image."""
    img_h, img_w = image.shape[:2]
    
    # Convert YOLO format to pixel coordinates
    x_center_px = int(x_center * img_w)
    y_center_px = int(y_center * img_h)
    box_w = int(width * img_w)
    box_h = int(height * img_h)
    
    # Calculate top-left corner
    x1 = int(x_center_px - box_w / 2)
    y1 = int(y_center_px - box_h / 2)
    x2 = int(x_center_px + box_w / 2)
    y2 = int(y_center_px + box_h / 2)
    
    # Draw box
    color = CLASS_COLORS.get(class_id, (255, 255, 255))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Draw label
    label = CLASS_NAMES.get(class_id, f'class_{class_id}')
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    
    # Background for label
    cv2.rectangle(image, (x1, y1 - label_size[1] - 5), 
                  (x1 + label_size[0], y1), color, -1)
    
    # Label text
    cv2.putText(image, label, (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return image

def visualize_annotations(dataset_dir, split='all', output_dir=None, sample=None):
    """
    Visualize annotated images.
    
    Args:
        dataset_dir: Dataset directory
        split: 'all', 'train', 'val', or 'none' (for pre-split data)
        output_dir: If provided, save images here instead of displaying
        sample: If set, only visualize N random images
    """
    dataset_dir = Path(dataset_dir)
    
    # Determine paths based on split
    if split == 'all':
        # Check if already split
        if (dataset_dir / 'images' / 'train').exists():
            images_dirs = [
                dataset_dir / 'images' / 'train',
                dataset_dir / 'images' / 'val'
            ]
            labels_dirs = [
                dataset_dir / 'labels' / 'train',
                dataset_dir / 'labels' / 'val'
            ]
        else:
            images_dirs = [dataset_dir / 'images']
            labels_dirs = [dataset_dir / 'labels']
    elif split in ['train', 'val']:
        images_dirs = [dataset_dir / 'images' / split]
        labels_dirs = [dataset_dir / 'labels' / split]
    else:  # 'none' - unsplit data
        images_dirs = [dataset_dir / 'images']
        labels_dirs = [dataset_dir / 'labels']
    
    # Collect all image-label pairs
    image_label_pairs = []
    for img_dir, lbl_dir in zip(images_dirs, labels_dirs):
        if not img_dir.exists():
            print(f"Warning: {img_dir} not found, skipping")
            continue
            
        for label_file in lbl_dir.glob('*.txt'):
            if label_file.name == 'classes.txt':
                continue
                
            if label_file.stat().st_size == 0:
                continue  # Skip empty labels
            
            # Find corresponding image
            img_file = None
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                test_file = img_dir / (label_file.stem + ext)
                if test_file.exists():
                    img_file = test_file
                    break
            
            if img_file:
                image_label_pairs.append((img_file, label_file))
    
    if not image_label_pairs:
        print(f"No annotated images found in {dataset_dir}")
        return
    
    print(f"Found {len(image_label_pairs)} annotated images")
    
    # Sample if requested
    if sample and sample < len(image_label_pairs):
        random.seed(42)
        image_label_pairs = random.sample(image_label_pairs, sample)
        print(f"Sampling {sample} images")
    
    # Setup output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving visualizations to: {output_dir}")
    
    # Process images
    stats = {'top_matra': 0, 'side_matra': 0, 'bottom_matra': 0}
    
    for img_path, label_path in tqdm(image_label_pairs, desc="Visualizing"):
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to read: {img_path}")
            continue
        
        # Read labels
        with open(label_path) as f:
            lines = f.readlines()
        
        # Draw boxes
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            image = draw_yolo_box(image, class_id, x_center, y_center, width, height)
            
            # Update stats
            class_name = CLASS_NAMES.get(class_id, 'unknown')
            stats[class_name] = stats.get(class_name, 0) + 1
        
        # Save or display
        if output_dir:
            out_path = output_dir / img_path.name
            cv2.imwrite(str(out_path), image)
        else:
            # Display
            cv2.imshow('Modi Matra Annotations (Press any key for next, Q to quit)', image)
            key = cv2.waitKey(0)
            if key == ord('q') or key == ord('Q'):
                break
    
    if not output_dir:
        cv2.destroyAllWindows()
    
    # Print statistics
    print(f"\n{'='*50}")
    print("Annotation Statistics:")
    print(f"{'='*50}")
    for class_name, count in sorted(stats.items()):
        print(f"  {class_name}: {count} boxes")
    print(f"  Total: {sum(stats.values())} boxes")
    print(f"{'='*50}")
    
    if output_dir:
        print(f"\n✓ Visualizations saved to: {output_dir.absolute()}")

def main():
    parser = argparse.ArgumentParser(
        description='Visualize Modi matra annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View all annotations interactively
  python visualize_annotations.py --dataset datasets/modi_sample
  
  # Save visualizations to folder
  python visualize_annotations.py --dataset datasets/modi_sample --output viz_output
  
  # View only training set
  python visualize_annotations.py --dataset datasets/modi_sample --split train
  
  # Sample 20 random images
  python visualize_annotations.py --dataset datasets/modi_sample --sample 20
        """
    )
    parser.add_argument('--dataset', required=True, help='Dataset directory')
    parser.add_argument('--split', default='all', 
                       choices=['all', 'train', 'val', 'none'],
                       help='Which split to visualize')
    parser.add_argument('--output', help='Output directory to save visualizations')
    parser.add_argument('--sample', type=int, help='Visualize N random images')
    args = parser.parse_args()
    
    if not Path(args.dataset).exists():
        print(f"ERROR: Dataset not found: {args.dataset}")
        return
    
    visualize_annotations(args.dataset, args.split, args.output, args.sample)

if __name__ == '__main__':
    main()
