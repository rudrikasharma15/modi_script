#!/usr/bin/env python3
"""
convert_labelme_to_yolo.py
Converts labelme JSON annotations to YOLO format.

Usage:
    python convert_labelme_to_yolo.py --input datasets/modi_300/labels_json --output datasets/modi_300/labels
"""
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# Class name to ID mapping
CLASS_MAPPING = {
    'top_matra': 0,
    'side_matra': 1,
    'bottom_matra': 2
}

def convert_labelme_to_yolo(json_file, output_dir):
    """Convert single labelme JSON to YOLO format."""
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get image dimensions
    img_width = data['imageWidth']
    img_height = data['imageHeight']
    
    # Output file
    output_file = output_dir / (Path(json_file).stem + '.txt')
    
    yolo_lines = []
    
    # Process each shape (annotation)
    for shape in data.get('shapes', []):
        label = shape['label']
        points = shape['points']
        
        # Skip if label not in our classes
        if label not in CLASS_MAPPING:
            print(f"Warning: Unknown label '{label}' in {json_file.name}")
            continue
        
        class_id = CLASS_MAPPING[label]
        
        # Get bounding box from points (x1, y1, x2, y2)
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        
        # Convert to YOLO format (normalized center x, center y, width, height)
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        # YOLO format: class_id x_center y_center width height
        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        yolo_lines.append(yolo_line)
    
    # Write to file
    if yolo_lines:
        with open(output_file, 'w') as f:
            f.write('\n'.join(yolo_lines))
        return True
    else:
        # Create empty file if no annotations
        output_file.touch()
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert labelme JSON to YOLO format')
    parser.add_argument('--input', required=True, help='Input directory with JSON files')
    parser.add_argument('--output', required=True, help='Output directory for YOLO txt files')
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files
    json_files = list(input_dir.glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    print(f"Converting to YOLO format...")
    
    converted = 0
    empty = 0
    
    for json_file in tqdm(json_files):
        try:
            has_annotations = convert_labelme_to_yolo(json_file, output_dir)
            if has_annotations:
                converted += 1
            else:
                empty += 1
        except Exception as e:
            print(f"Error converting {json_file.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"{'='*60}")
    print(f"Total files processed: {len(json_files)}")
    print(f"Files with annotations: {converted}")
    print(f"Empty files: {empty}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()