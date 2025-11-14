#!/usr/bin/env python3
"""
prepare_modi_dataset.py
Prepares Modi character images for manual matra annotation.

Usage:
    # Start with sample
    python prepare_modi_dataset.py --data_root "/path/to/Dataset_Modi" --out datasets/modi_matra_sample --sample 100
    
    # Full dataset
    python prepare_modi_dataset.py --data_root "/path/to/Dataset_Modi" --out datasets/modi_matra_full
"""
import os
import argparse
from pathlib import Path
import shutil
from tqdm import tqdm
import random
import json

# Matra type mapping for reference during annotation
MATRA_REFERENCE = {
    'top_matra': ['i', 'e', 'ai', 'nm', 'am'],  # Appears above
    'side_matra': ['aa', 'o', 'au', 'ahaa', 'aha', 'ah'],  # Appears on side
    'bottom_matra': ['u', 'uu'],  # Appears below
}

def parse_folder_name(folder_name):
    """Extract matra info from folder name for reference."""
    parts = folder_name.split()
    if len(parts) < 2:
        return None, None
    
    char_part = parts[1].split('-')[0].upper()
    
    # Base consonants
    base_consonants = ['KSH', 'DNYA', 'DNY', 'KH', 'GH', 'CHH', 'TR', 'TT', 
                       'DH', 'DHH', 'TH', 'THH', 'BH', 'PH', 'SH', 'AL',
                       'K', 'G', 'CH', 'J', 'Z', 'D', 'N', 'T', 'B', 
                       'M', 'Y', 'R', 'L', 'V', 'S', 'H', 'P']
    
    # Find base (check longer consonants first)
    base = None
    for cons in base_consonants:
        if char_part.startswith(cons):
            base = cons
            break
    
    if not base:
        # Check if it's a pure vowel (no matra to detect)
        vowels = ['A', 'AA', 'I', 'II', 'U', 'UU', 'E', 'AI', 'O', 'AU', 'NM', 'AHAA']
        if char_part in vowels:
            return None, None  # Pure vowel, no matra
        base = char_part[0] if char_part else None
    
    if not base:
        return None, None
    
    # Extract matra part
    matra_part = char_part[len(base):].lower()
    
    if not matra_part or matra_part == 'a':
        return None, None  # No matra or base 'a' sound
    
    # Determine matra type
    for matra_type, variants in MATRA_REFERENCE.items():
        if matra_part in variants:
            return matra_part, matra_type
    
    return matra_part, 'unknown'

def prepare_dataset(data_root, out_dir, train_split=0.85, sample_size=None):
    """
    Prepare dataset for annotation.
    
    Args:
        data_root: Root directory with character folders
        out_dir: Output directory
        train_split: Fraction for training (not used yet, for future)
        sample_size: If set, randomly sample this many images per matra type
    """
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    
    # Create directories
    images_dir = out_dir / 'images'
    labels_dir = out_dir / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all folders
    folders = [f for f in data_root.iterdir() if f.is_dir()]
    
    img_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    
    # Collect images by matra type
    images_by_matra = {'top_matra': [], 'side_matra': [], 'bottom_matra': []}
    skipped = []
    unknown = []
    
    print("Scanning folders...")
    for folder in tqdm(folders, desc="Scanning"):
        matra_part, matra_type = parse_folder_name(folder.name)
        
        if matra_type is None:
            skipped.append(folder.name)
            continue
        
        if matra_type == 'unknown':
            unknown.append(folder.name)
            continue
        
        # Find images
        images = [f for f in folder.iterdir() 
                 if f.is_file() and f.suffix.lower() in img_extensions]
        
        for img_path in images:
            images_by_matra[matra_type].append({
                'path': img_path,
                'folder': folder.name,
                'matra': matra_part
            })
    
    # Print statistics
    print("\nDataset statistics:")
    for matra_type, images in images_by_matra.items():
        print(f"  {matra_type}: {len(images)} images")
    print(f"  Skipped (no matra): {len(skipped)} folders")
    if unknown:
        print(f"  Unknown matra type: {len(unknown)} folders")
        print(f"    Examples: {unknown[:5]}")
    
    # Sample if requested
    if sample_size:
        print(f"\nSampling {sample_size} images per matra type...")
        random.seed(42)
        for matra_type in images_by_matra:
            available = len(images_by_matra[matra_type])
            if available > sample_size:
                images_by_matra[matra_type] = random.sample(
                    images_by_matra[matra_type], sample_size
                )
            else:
                print(f"  Warning: Only {available} {matra_type} images available (requested {sample_size})")
    
    # Copy images and create metadata
    metadata = []
    total_copied = 0
    
    print("\nCopying images...")
    for matra_type, images in images_by_matra.items():
        for img_info in tqdm(images, desc=f"Processing {matra_type}"):
            src = img_info['path']
            # Create unique filename
            dest_name = f"{img_info['folder'].replace(' ', '_')}_{src.name}"
            dest = images_dir / dest_name
            
            try:
                shutil.copy2(src, dest)
                
                # Create empty label file
                label_file = labels_dir / (dest.stem + '.txt')
                label_file.touch()
                
                metadata.append({
                    'image': dest.name,
                    'matra_type': matra_type,
                    'matra': img_info['matra'],
                    'original_folder': img_info['folder']
                })
                total_copied += 1
            except Exception as e:
                print(f"Error copying {src}: {e}")
    
    # Save metadata
    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create annotation guide
    guide_content = f"""# Modi Matra Annotation Guide

## Dataset Information
Total images prepared: {total_copied}
- Top matras: {len([m for m in metadata if m['matra_type'] == 'top_matra'])}
- Side matras: {len([m for m in metadata if m['matra_type'] == 'side_matra'])}
- Bottom matras: {len([m for m in metadata if m['matra_type'] == 'bottom_matra'])}

Skipped folders (no matras): {len(skipped)}

## Classes (YOLO Format)
0 - top_matra (appears above base character)
1 - side_matra (appears to the right/side of base)
2 - bottom_matra (appears below base character)

## Setup LabelImg

1. Install LabelImg:
   ```
   pip install labelImg
   ```

2. Start annotation:
   ```
   labelImg {images_dir.absolute()} {labels_dir.absolute()}
   ```

3. In LabelImg settings:
   - View > Auto Save Mode (recommended)
   - Press Ctrl+Y to switch to YOLO format
   - File > Change Save Dir > Select: {labels_dir.absolute()}

## Annotation Instructions

### How to Annotate Each Image

1. **Understand what you see**: Each image shows a complete Modi character (base + matra)

2. **Identify the matra**: Look for the vowel mark that modifies the base consonant
   - Check metadata.json for hints about expected matra type
   
3. **Draw bounding box**: Press 'W' key
   - Draw box ONLY around the matra part
   - Do NOT include the base consonant in the box
   - Keep box tight around the matra strokes

4. **Select class**: Choose the correct class number
   - 0 = top_matra (marks above)
   - 1 = side_matra (marks on right side)
   - 2 = bottom_matra (marks below)

5. **Save and continue**:
   - Press Ctrl+S to save (or auto-save if enabled)
   - Press D to go to next image

### Matra Examples

**Top Matras (Class 0):**
- i (ि) - Short vertical stroke above
- e (े) - Horizontal line above
- ai (ै) - Two marks above
- nm/am (ं) - Dot/circle above (anusvara)

**Side Matras (Class 1):**
- aa (ा) - Vertical line to the right
- o (ो) - Curved mark on right
- au (ौ) - Additional marks on right
- ahaa (ः) - Two dots on right (visarga)

**Bottom Matras (Class 2):**
- u (ु) - Small mark below
- uu (ू) - Longer mark below

## Annotation Tips

✓ **Focus only on matras** - Ignore the base consonant completely
✓ **Tight boxes** - Boxes should closely fit the matra strokes
✓ **Multiple parts** - If a matra has separate parts (e.g., top + side), create separate boxes
✓ **Connected writing** - For flowing/cursive Modi, use your best judgment for boundaries
✓ **Consistency** - Try to annotate similar matras the same way across images
✓ **When in doubt** - Check the metadata.json for the expected matra type

## Quality Checks

Before finishing, verify:
- [ ] All label files have content (not empty)
- [ ] Boxes are around matras, not whole characters
- [ ] Class labels are correct (0=top, 1=side, 2=bottom)
- [ ] No boxes overlap incorrectly
- [ ] Similar matras have similar annotations

## After Annotation

Once you've annotated all images:

1. Run the split creation script:
   ```
   python create_splits.py --dataset {out_dir.absolute()}
   ```

2. Train the model:
   ```
   yolo detect train data={out_dir.absolute()}/modi_matra.yaml model=yolov8n.pt epochs=100
   ```

## Keyboard Shortcuts in LabelImg

- W - Create bounding box
- D - Next image
- A - Previous image
- Del - Delete selected box
- Ctrl+S - Save
- Ctrl+D - Duplicate box
- Space - Flag image as verified

## Need Help?

- Refer to metadata.json to see expected matra type for each image
- Start with a few images to establish your annotation style
- Take breaks to maintain consistency
- Review your first 20-30 annotations before continuing
"""
    
    with open(out_dir / 'ANNOTATION_GUIDE.md', 'w') as f:
        f.write(guide_content)
    
    # Create classes.txt for LabelImg
    with open(out_dir / 'classes.txt', 'w') as f:
        f.write("top_matra\nside_matra\nbottom_matra\n")
    
    # Create predefined_classes.txt (LabelImg looks for this)
    with open(labels_dir / 'classes.txt', 'w') as f:
        f.write("top_matra\nside_matra\nbottom_matra\n")
    
    print(f"\n{'='*70}")
    print(f"✓ Dataset prepared successfully!")
    print(f"{'='*70}")
    print(f"\nLocation: {out_dir.absolute()}")
    print(f"Total images: {total_copied}")
    print(f"\n📋 NEXT STEPS:")
    print(f"\n1. Read the annotation guide:")
    print(f"   cat {out_dir}/ANNOTATION_GUIDE.md")
    print(f"\n2. Install LabelImg:")
    print(f"   pip install labelImg")
    print(f"\n3. Start annotating:")
    print(f"   labelImg {images_dir.absolute()} {labels_dir.absolute()}")
    print(f"\n4. After annotation, create train/val splits:")
    print(f"   python create_splits.py --dataset {out_dir.absolute()}")
    print(f"\n{'='*70}")
    
    return total_copied

def main():
    parser = argparse.ArgumentParser(
        description='Prepare Modi script images for matra annotation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with 100 images per matra type (recommended)
  python prepare_modi_dataset.py --data_root ./Dataset_Modi --out datasets/modi_sample --sample 100
  
  # Prepare full dataset
  python prepare_modi_dataset.py --data_root ./Dataset_Modi --out datasets/modi_full
        """
    )
    parser.add_argument('--data_root', required=True, help='Root directory with character folders')
    parser.add_argument('--out', default='datasets/modi_matra', help='Output directory')
    parser.add_argument('--sample', type=int, help='Sample N images per matra type (for testing)')
    parser.add_argument('--train_split', type=float, default=0.85, help='Train split ratio (unused for now)')
    args = parser.parse_args()
    
    if not Path(args.data_root).exists():
        print(f"ERROR: Data root not found: {args.data_root}")
        return
    
    prepare_dataset(args.data_root, args.out, args.train_split, args.sample)

if __name__ == '__main__':
    main()
