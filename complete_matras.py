#!/usr/bin/env python3
"""
autolabel_matra_folders.py
Generate YOLO labels for Modi script matras using folder names.

Usage:
    python autolabel_matra_folders.py --data_root "/Users/applemaair/Downloads/Dataset_Modi/Dataset_Modi" --out datasets/modi
"""
import os
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import json

# Matra mapping based on Modi script vowel signs
MATRA_RULES = {
    # Top matras (appear above base character)
    'i': 'top_matra',      # ि
    'e': 'top_matra',      # े  
    'ai': 'top_matra',     # ै
    
    # Side matras (appear to right/inline)
    'aa': 'side_matra',    # ा
    'o': 'side_matra',     # ो
    'au': 'side_matra',    # ौ
    
    # Bottom matras (appear below base character)
    'u': 'bottom_matra',   # ु
    
    # Anusvara/Visarga (often top, but sometimes separate)
    'nm': 'top_matra',     # ं (anusvara - appears above)
    'ahaa': 'side_matra',  # ः (visarga - appears to side)
    'aha': 'side_matra',
    'am': 'top_matra',
    'ah': 'side_matra',
}

# Class to index mapping
CLASS_TO_IDX = {
    'top_matra': 0,
    'side_matra': 1,
    'bottom_matra': 2
}

def parse_folder_name(folder_name):
    """
    Extract matra type from folder name.
    Examples:
        '13 KI-kiran' -> 'i' -> 'top_matra'
        '14 KU-kunfu' -> 'u' -> 'bottom_matra'
        '15 KE-kedar' -> 'e' -> 'top_matra'
    """
    # Remove leading numbers and split
    parts = folder_name.split()
    if len(parts) < 2:
        return None
    
    # Get the character part (e.g., "KI-kiran")
    char_part = parts[1].split('-')[0].upper()
    
    # Extract matra from character combination
    # Base consonants in Modi: KA, KHA, GA, etc.
    base_consonants = ['K', 'KH', 'G', 'GH', 'CH', 'CHH', 'J', 'Z', 
                       'TR', 'TT', 'D', 'DH', 'DHH', 'N', 'T', 'TH', 'THH',
                       'B', 'BH', 'M', 'Y', 'R', 'L', 'V', 'SH', 'S', 'H',
                       'P', 'PH', 'AL', 'KSH', 'DNY', 'DNYA']
    
    # Find base consonant
    base = None
    for cons in sorted(base_consonants, key=len, reverse=True):
        if char_part.startswith(cons):
            base = cons
            break
    
    if not base:
        # Check for standalone vowels (no matra needed)
        vowels = ['A', 'AA', 'I', 'U', 'E', 'AI', 'O', 'AU']
        if char_part in vowels or any(char_part.startswith(v) for v in vowels):
            return None
        base = char_part[0]  # Fallback
    
    # Extract matra part
    matra_part = char_part[len(base):].lower()
    
    if not matra_part or matra_part == 'a':
        return None  # No matra (inherent 'a')
    
    # Map to matra type
    return MATRA_RULES.get(matra_part)

def get_smart_bbox(img, matra_type):
    """
    Generate bounding box based on matra type and image analysis.
    Uses adaptive approach based on typical Modi matra positions.
    """
    h, w = img.shape[:2]
    
    # Simple binarization to find content bounds
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find content bounds
    coords = cv2.findNonZero(binary)
    if coords is None:
        # Fallback: center box
        return [0.5, 0.5, 0.6, 0.6]
    
    x, y, ww, hh = cv2.boundingRect(coords)
    
    # Calculate normalized values
    cx_full = (x + ww/2) / w
    cy_full = (y + hh/2) / h
    nw_full = ww / w
    nh_full = hh / h
    
    # Adjust based on matra type
    if matra_type == 'top_matra':
        # Focus on top portion
        cy = (y + hh * 0.25) / h  # Top quarter
        nh = (hh * 0.4) / h  # Upper 40%
        return [cx_full, cy, nw_full * 0.8, nh]
    
    elif matra_type == 'bottom_matra':
        # Focus on bottom portion
        cy = (y + hh * 0.75) / h  # Bottom quarter
        nh = (hh * 0.4) / h  # Lower 40%
        return [cx_full, cy, nw_full * 0.8, nh]
    
    elif matra_type == 'side_matra':
        # Focus on right side or middle-right
        cx = (x + ww * 0.65) / w  # Right-biased
        return [cx, cy_full, nw_full * 0.5, nh_full * 0.8]
    
    # Default: return full content box
    return [cx_full, cy_full, nw_full, nh_full]

def process_folder(folder_path, out_img_dir, out_label_dir):
    """Process all images in a folder."""
    folder_name = folder_path.name
    matra_type = parse_folder_name(folder_name)
    
    if matra_type is None:
        return 0, 0  # Skip folders without matras
    
    class_idx = CLASS_TO_IDX[matra_type]
    
    # Find all images
    img_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    images = [f for f in folder_path.iterdir() 
              if f.is_file() and f.suffix.lower() in img_extensions]
    
    success = 0
    for img_path in images:
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # Generate bbox
            cx, cy, nw, nh = get_smart_bbox(img, matra_type)
            
            # Clamp values
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.01, min(1.0, nw))
            nh = max(0.01, min(1.0, nh))
            
            # Copy image
            dest_img = out_img_dir / img_path.name
            from shutil import copyfile
            copyfile(str(img_path), str(dest_img))
            
            # Write label
            label_path = out_label_dir / (img_path.stem + '.txt')
            with open(label_path, 'w') as f:
                f.write(f"{class_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
            
            success += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return len(images), success

def main():
    parser = argparse.ArgumentParser(
        description='Auto-generate YOLO labels for Modi matras from folder structure'
    )
    parser.add_argument('--data_root', required=True, 
                       help='Root directory with character folders')
    parser.add_argument('--out', default='datasets/modi',
                       help='Output directory for YOLO dataset')
    parser.add_argument('--train_split', type=float, default=0.85,
                       help='Fraction of data for training (rest for validation)')
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    out = Path(args.out)
    
    # Create output directories
    train_img = out / 'images' / 'train'
    train_lbl = out / 'labels' / 'train'
    val_img = out / 'images' / 'val'
    val_lbl = out / 'labels' / 'val'
    
    for d in [train_img, train_lbl, val_img, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Get all character folders
    folders = [f for f in data_root.iterdir() if f.is_dir()]
    print(f"Found {len(folders)} character folders")
    
    # Process each folder
    total_imgs = 0
    total_success = 0
    stats = {'top_matra': 0, 'side_matra': 0, 'bottom_matra': 0}
    
    for folder in tqdm(folders, desc="Processing folders"):
        folder_name = folder.name
        matra_type = parse_folder_name(folder_name)
        
        if matra_type is None:
            continue
        
        # Decide train/val split at folder level for better distribution
        import random
        random.seed(42)  # Consistent splits
        is_train = random.random() < args.train_split
        
        img_dir = train_img if is_train else val_img
        lbl_dir = train_lbl if is_train else val_lbl
        
        total, success = process_folder(folder, img_dir, lbl_dir)
        total_imgs += total
        total_success += success
        stats[matra_type] += success
    
    # Create YAML config
    yaml_content = f"""# Modi Matra Detection Dataset
path: {out.absolute()}
train: images/train
val: images/val

nc: 3
names: ['top_matra', 'side_matra', 'bottom_matra']

# Dataset statistics
# Total images: {total_success}
# Top matras: {stats['top_matra']}
# Side matras: {stats['side_matra']}
# Bottom matras: {stats['bottom_matra']}
"""
    
    yaml_path = out / 'modi_matra.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n{'='*60}")
    print(f"Dataset created successfully!")
    print(f"{'='*60}")
    print(f"Total images processed: {total_success}/{total_imgs}")
    print(f"\nClass distribution:")
    print(f"  Top matras:    {stats['top_matra']}")
    print(f"  Side matras:   {stats['side_matra']}")
    print(f"  Bottom matras: {stats['bottom_matra']}")
    print(f"\nDataset YAML: {yaml_path}")
    print(f"\nTo train:")
    print(f"  yolo detect train data={yaml_path} model=yolov8n.pt epochs=50 imgsz=640")
    
if __name__ == '__main__':
    main()
