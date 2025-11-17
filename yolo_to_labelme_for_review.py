#!/usr/bin/env python3
"""
Convert YOLO format labels to labelme JSON format for easy review.
This allows viewing model predictions as pre-drawn boxes in labelme.
"""
import argparse
import json
import base64
from pathlib import Path
from PIL import Image
import io

def yolo_to_labelme(images_dir, labels_dir, output_dir, class_names):
    """Convert YOLO labels to labelme JSON format."""
    
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Images: {images_dir}")
    print(f"📂 Labels: {labels_dir}")
    print(f"📂 Output: {output_dir}\n")
    
    # Find all images
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_files.extend(list(images_dir.glob(ext)))
    
    print(f"Found {len(image_files)} images")
    
    converted = 0
    skipped = 0
    
    for img_path in image_files:
        # Find corresponding label file
        label_path = labels_dir / (img_path.stem + '.txt')
        
        if not label_path.exists():
            skipped += 1
            continue
        
        # Load image to get dimensions
        try:
            img = Image.open(img_path)
            img_width, img_height = img.size
        except Exception as e:
            print(f"⚠️  Error loading {img_path.name}: {e}")
            skipped += 1
            continue
        
        # Read image as base64 (optional, for embedding in JSON)
        # For large datasets, you can skip this and set imageData to None
        with open(img_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Parse YOLO labels
        shapes = []
        with open(label_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert normalized coordinates to pixel coordinates
                    x_center_px = x_center * img_width
                    y_center_px = y_center * img_height
                    width_px = width * img_width
                    height_px = height * img_height
                    
                    # Calculate corner coordinates
                    x1 = x_center_px - width_px / 2
                    y1 = y_center_px - height_px / 2
                    x2 = x_center_px + width_px / 2
                    y2 = y_center_px + height_px / 2
                    
                    # Create labelme shape
                    shape = {
                        "label": class_names[class_id],
                        "points": [
                            [x1, y1],
                            [x2, y2]
                        ],
                        "group_id": None,
                        "shape_type": "rectangle",
                        "flags": {}
                    }
                    shapes.append(shape)
        
        # Create labelme JSON
        labelme_data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": img_path.name,
            "imageData": image_data,
            "imageHeight": img_height,
            "imageWidth": img_width
        }
        
        # Save JSON
        json_path = output_dir / (img_path.stem + '.json')
        with open(json_path, 'w') as f:
            json.dump(labelme_data, f, indent=2)
        
        converted += 1
    
    print(f"\n✅ Converted: {converted} images")
    print(f"⊘ Skipped: {skipped} images (no labels)")
    print(f"\n📁 Output saved to: {output_dir}")
    print(f"\n📝 Now you can review with:")
    print(f"labelme {output_dir} --labels labels.txt --nodata")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert YOLO labels to labelme JSON for review')
    parser.add_argument('--images', required=True, help='Directory containing images')
    parser.add_argument('--labels', required=True, help='Directory containing YOLO label files')
    parser.add_argument('--output', required=True, help='Output directory for labelme JSON files')
    parser.add_argument('--class_names', nargs='+', default=['top_matra', 'side_matra', 'bottom_matra'],
                        help='Class names in order')
    
    args = parser.parse_args()
    
    yolo_to_labelme(args.images, args.labels, args.output, args.class_names)