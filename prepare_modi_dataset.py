# #!/usr/bin/env python3
# """
# prepare_modi_dataset.py
# Prepares Modi character images for manual matra annotation.

# Usage:
#     # Start with sample
#     python prepare_modi_dataset.py --data_root "/path/to/Dataset_Modi" --out datasets/modi_matra_sample --sample 100
    
#     # Full dataset
#     python prepare_modi_dataset.py --data_root "/path/to/Dataset_Modi" --out datasets/modi_matra_full
# """
# import os
# import argparse
# from pathlib import Path
# import shutil
# from tqdm import tqdm
# import random
# import json

# # Matra type mapping for reference during annotation
# MATRA_REFERENCE = {
#     'top_matra': ['i', 'e', 'ai', 'n', 'm', 'am', 'nm'],  # Appears above
#     'side_matra': ['aa', 'o', 'au', 'ao', 'aha', 'ahaa'],  # Appears on side
#     'bottom_matra': ['u', 'uu'],  # Appears below
# }

# def parse_folder_name(folder_name):
#     """Extract matra info from folder name for reference."""
#     parts = folder_name.split()
#     if len(parts) < 2:
#         return None, None
    
#     char_part = parts[1].split('-')[0].upper()
    
#     # Base consonants
#     base_consonants = ['KSH', 'DNYA', 'DNY', 'KH', 'GH', 'CHH', 'TR', 'TT', 
#                        'DH', 'DHH', 'TH', 'THH', 'BH', 'PH', 'SH', 'AL',
#                        'K', 'G', 'CH', 'J', 'Z', 'D', 'N', 'T', 'B', 
#                        'M', 'Y', 'R', 'L', 'V', 'S', 'H', 'P']
    
#     # Find base (check longer consonants first)
#     base = None
#     for cons in base_consonants:
#         if char_part.startswith(cons):
#             base = cons
#             break
    
#     if not base:
#         # Check if it's a pure vowel (no matra to detect)
#         vowels = ['A', 'AA', 'I', 'II', 'U', 'UU', 'E', 'AI', 'O', 'AU', 'NM', 'AHAA']
#         if char_part in vowels:
#             return None, None  # Pure vowel, no matra
#         base = char_part[0] if char_part else None
    
#     if not base:
#         return None, None
    
#     # Extract matra part
#     matra_part = char_part[len(base):].lower()
    
#     if not matra_part or matra_part == 'a':
#         return None, None  # No matra or base 'a' sound
    
#     # Determine matra type
#     for matra_type, variants in MATRA_REFERENCE.items():
#         if matra_part in variants:
#             return matra_part, matra_type
    
#     return matra_part, 'unknown'

# def prepare_dataset(data_root, out_dir, train_split=0.85, sample_size=None):
#     """
#     Prepare dataset for annotation.
    
#     Args:
#         data_root: Root directory with character folders
#         out_dir: Output directory
#         train_split: Fraction for training (not used yet, for future)
#         sample_size: If set, randomly sample this many images per matra type
#     """
#     data_root = Path(data_root)
#     out_dir = Path(out_dir)
    
#     # Create directories
#     images_dir = out_dir / 'images'
#     labels_dir = out_dir / 'labels'
#     images_dir.mkdir(parents=True, exist_ok=True)
#     labels_dir.mkdir(parents=True, exist_ok=True)
    
#     # Get all folders
#     folders = [f for f in data_root.iterdir() if f.is_dir()]
    
#     img_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    
#     # Collect images by matra type
#     images_by_matra = {'top_matra': [], 'side_matra': [], 'bottom_matra': []}
#     skipped = []
#     unknown = []
    
#     print("Scanning folders...")
#     for folder in tqdm(folders, desc="Scanning"):
#         matra_part, matra_type = parse_folder_name(folder.name)
        
#         if matra_type is None:
#             skipped.append(folder.name)
#             continue
        
#         if matra_type == 'unknown':
#             unknown.append(folder.name)
#             continue
        
#         # Find images
#         images = [f for f in folder.iterdir() 
#                  if f.is_file() and f.suffix.lower() in img_extensions]
        
#         for img_path in images:
#             images_by_matra[matra_type].append({
#                 'path': img_path,
#                 'folder': folder.name,
#                 'matra': matra_part
#             })
    
#     # Print statistics
#     print("\nDataset statistics:")
#     for matra_type, images in images_by_matra.items():
#         print(f"  {matra_type}: {len(images)} images")
#     print(f"  Skipped (no matra): {len(skipped)} folders")
#     if unknown:
#         print(f"  Unknown matra type: {len(unknown)} folders")
#         print(f"    Examples: {unknown[:5]}")
    
#     # Sample if requested
#     if sample_size:
#         print(f"\nSampling {sample_size} images per matra type...")
#         random.seed(42)
#         for matra_type in images_by_matra:
#             available = len(images_by_matra[matra_type])
#             if available > sample_size:
#                 images_by_matra[matra_type] = random.sample(
#                     images_by_matra[matra_type], sample_size
#                 )
#             else:
#                 print(f"  Warning: Only {available} {matra_type} images available (requested {sample_size})")
    
#     # Copy images and create metadata
#     metadata = []
#     total_copied = 0
    
#     print("\nCopying images...")
#     for matra_type, images in images_by_matra.items():
#         for img_info in tqdm(images, desc=f"Processing {matra_type}"):
#             src = img_info['path']
#             # Create unique filename
#             dest_name = f"{img_info['folder'].replace(' ', '_')}_{src.name}"
#             dest = images_dir / dest_name
            
#             try:
#                 shutil.copy2(src, dest)
                
#                 # Create empty label file
#                 label_file = labels_dir / (dest.stem + '.txt')
#                 label_file.touch()
                
#                 metadata.append({
#                     'image': dest.name,
#                     'matra_type': matra_type,
#                     'matra': img_info['matra'],
#                     'original_folder': img_info['folder']
#                 })
#                 total_copied += 1
#             except Exception as e:
#                 print(f"Error copying {src}: {e}")
    
#     # Save metadata
#     with open(out_dir / 'metadata.json', 'w', encoding='utf-8') as f:
#         json.dump(metadata, f, indent=2)
    
#     # Create annotation guide (with UTF-8 encoding fix)
#     guide_content = f"""# Modi Matra Annotation Guide

# ## Dataset Information
# Total images prepared: {total_copied}
# - Top matras: {len([m for m in metadata if m['matra_type'] == 'top_matra'])}
# - Side matras: {len([m for m in metadata if m['matra_type'] == 'side_matra'])}
# - Bottom matras: {len([m for m in metadata if m['matra_type'] == 'bottom_matra'])}

# Skipped folders (no matras): {len(skipped)}

# ## Classes (YOLO Format)
# 0 - top_matra (appears above base character)
# 1 - side_matra (appears to the right/side of base)
# 2 - bottom_matra (appears below base character)

# ## Setup LabelImg

# 1. Install LabelImg:
#    ```
#    pip install labelImg
#    ```

# 2. Start annotation:
#    ```
#    labelImg {images_dir.absolute()} {labels_dir.absolute()}
#    ```

# 3. In LabelImg settings:
#    - View > Auto Save Mode (recommended)
#    - Press Ctrl+Y to switch to YOLO format
#    - File > Change Save Dir > Select: {labels_dir.absolute()}

# ## Annotation Instructions

# ### How to Annotate Each Image

# 1. **Understand what you see**: Each image shows a complete Modi character (base + matra)

# 2. **Identify the matra**: Look for the vowel mark that modifies the base consonant
#    - Check metadata.json for hints about expected matra type
   
# 3. **Draw bounding box**: Press 'W' key
#    - Draw box ONLY around the matra part
#    - Do NOT include the base consonant in the box
#    - Keep box tight around the matra strokes

# 4. **Select class**: Choose the correct class number
#    - 0 = top_matra (marks above)
#    - 1 = side_matra (marks on right side)
#    - 2 = bottom_matra (marks below)

# 5. **Save and continue**:
#    - Press Ctrl+S to save (or auto-save if enabled)
#    - Press D to go to next image

# ### Matra Examples

# **Top Matras (Class 0):**
# - i (ि) - Short vertical stroke above
# - e (े) - Horizontal line above
# - ai (ै) - Two marks above
# - nm/am (ं) - Dot/circle above (anusvara)

# **Side Matras (Class 1):**
# - aa (ा) - Vertical line to the right
# - o (ो) - Curved mark on right
# - au (ौ) - Additional marks on right
# - ahaa (ः) - Two dots on right (visarga)

# **Bottom Matras (Class 2):**
# - u (ु) - Small mark below
# - uu (ू) - Longer mark below

# ## Annotation Tips

# ✓ **Focus only on matras** - Ignore the base consonant completely
# ✓ **Tight boxes** - Boxes should closely fit the matra strokes
# ✓ **Multiple parts** - If a matra has separate parts (e.g., top + side), create separate boxes
# ✓ **Connected writing** - For flowing/cursive Modi, use your best judgment for boundaries
# ✓ **Consistency** - Try to annotate similar matras the same way across images
# ✓ **When in doubt** - Check the metadata.json for the expected matra type

# ## Quality Checks

# Before finishing, verify:
# - [ ] All label files have content (not empty)
# - [ ] Boxes are around matras, not whole characters
# - [ ] Class labels are correct (0=top, 1=side, 2=bottom)
# - [ ] No boxes overlap incorrectly
# - [ ] Similar matras have similar annotations

# ## After Annotation

# Once you've annotated all images:

# 1. Run the split creation script:
#    ```
#    python create_splits.py --dataset {out_dir.absolute()}
#    ```

# 2. Train the model:
#    ```
#    yolo detect train data={out_dir.absolute()}/modi_matra.yaml model=yolov8n.pt epochs=100
#    ```

# ## Keyboard Shortcuts in LabelImg

# - W - Create bounding box
# - D - Next image
# - A - Previous image
# - Del - Delete selected box
# - Ctrl+S - Save
# - Ctrl+D - Duplicate box
# - Space - Flag image as verified

# ## Need Help?

# - Refer to metadata.json to see expected matra type for each image
# - Start with a few images to establish your annotation style
# - Take breaks to maintain consistency
# - Review your first 20-30 annotations before continuing
# """
    
#     with open(out_dir / 'ANNOTATION_GUIDE.md', 'w', encoding='utf-8') as f:
#         f.write(guide_content)
    
#     # Create classes.txt for LabelImg
#     with open(out_dir / 'classes.txt', 'w') as f:
#         f.write("top_matra\nside_matra\nbottom_matra\n")
    
#     # Create predefined_classes.txt (LabelImg looks for this)
#     with open(labels_dir / 'classes.txt', 'w') as f:
#         f.write("top_matra\nside_matra\nbottom_matra\n")
    
#     print(f"\n{'='*70}")
#     print(f"✓ Dataset prepared successfully!")
#     print(f"{'='*70}")
#     print(f"\nLocation: {out_dir.absolute()}")
#     print(f"Total images: {total_copied}")
#     print(f"\n📋 NEXT STEPS:")
#     print(f"\n1. Read the annotation guide:")
#     print(f"   cat {out_dir}/ANNOTATION_GUIDE.md")
#     print(f"\n2. Install LabelImg:")
#     print(f"   pip install labelImg")
#     print(f"\n3. Start annotating:")
#     print(f"   labelImg {images_dir.absolute()} {labels_dir.absolute()}")
#     print(f"\n4. After annotation, create train/val splits:")
#     print(f"   python create_splits.py --dataset {out_dir.absolute()}")
#     print(f"\n{'='*70}")
    
#     return total_copied

# def main():
#     parser = argparse.ArgumentParser(
#         description='Prepare Modi script images for matra annotation',
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   # Start with 100 images per matra type (recommended)
#   python prepare_modi_dataset.py --data_root ./Dataset_Modi --out datasets/modi_sample --sample 100
  
#   # Prepare full dataset
#   python prepare_modi_dataset.py --data_root ./Dataset_Modi --out datasets/modi_full
#         """
#     )
#     parser.add_argument('--data_root', required=True, help='Root directory with character folders')
#     parser.add_argument('--out', default='datasets/modi_matra', help='Output directory')
#     parser.add_argument('--sample', type=int, help='Sample N images per matra type (for testing)')
#     parser.add_argument('--train_split', type=float, default=0.85, help='Train split ratio (unused for now)')
#     args = parser.parse_args()
    
#     if not Path(args.data_root).exists():
#         print(f"ERROR: Data root not found: {args.data_root}")
#         return
    
#     prepare_dataset(args.data_root, args.out, args.train_split, args.sample)

# if __name__ == '__main__':
#     main()
#!/usr/bin/env python3
"""
prepare_modi_dataset.py - Enhanced with targeted matra selection
Prepares Modi character images for manual matra annotation.

Usage:
    # Balanced sample (default)
    python prepare_modi_dataset.py --data_root "/path/to/Dataset_Modi" --out datasets/modi_sample --sample 100
    
    # ONLY side matras (for boosting weak class)
    python prepare_modi_dataset.py --data_root "/path/to/Dataset_Modi" --out datasets/modi_side_only --sample 100 --matra_types side
    
    # Side + Bottom matras only
    python prepare_modi_dataset.py --data_root "/path/to/Dataset_Modi" --out datasets/modi_side_bottom --sample 50 --matra_types side,bottom
    
    # Full dataset (all types)
    python prepare_modi_dataset.py --data_root "/path/to/Dataset_Modi" --out datasets/modi_full
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
    'top_matra': ['i', 'e', 'ai', 'n', 'm', 'am', 'nm'],  # Appears above
    'side_matra': ['aa', 'o', 'au', 'ao', 'aha', 'ahaa'],  # Appears on side
    'bottom_matra': ['u', 'uu'],  # Appears below
}

def parse_folder_name(folder_name):
    """Extract matra info from folder name for reference."""
    parts = folder_name.split()
    if len(parts) < 2:
        return None, None
    
    char_part = parts[1].split('-')[0].upper()
    
    # Base consonants (check longer ones first)
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

def prepare_dataset(data_root, out_dir, train_split=0.85, sample_size=None, target_matra_types=None):
    """
    Prepare dataset for annotation.
    
    Args:
        data_root: Root directory with character folders
        out_dir: Output directory
        train_split: Fraction for training (not used yet, for future)
        sample_size: If set, randomly sample this many images per matra type
        target_matra_types: List of matra types to include ['top', 'side', 'bottom']
                           If None, includes all types
    """
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    
    # Parse target matra types
    if target_matra_types:
        matra_type_map = {
            'top': 'top_matra',
            'side': 'side_matra', 
            'bottom': 'bottom_matra'
        }
        target_types = []
        for t in target_matra_types:
            if t in matra_type_map:
                target_types.append(matra_type_map[t])
            elif t in ['top_matra', 'side_matra', 'bottom_matra']:
                target_types.append(t)
        
        if not target_types:
            print("ERROR: Invalid matra types specified!")
            print("Valid options: top, side, bottom (or top_matra, side_matra, bottom_matra)")
            return 0
    else:
        target_types = ['top_matra', 'side_matra', 'bottom_matra']
    
    print(f"\n🎯 Target matra types: {', '.join(target_types)}")
    
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
    
    print("\n📂 Scanning folders...")
    for folder in tqdm(folders, desc="Scanning"):
        matra_part, matra_type = parse_folder_name(folder.name)
        
        if matra_type is None:
            skipped.append(folder.name)
            continue
        
        if matra_type == 'unknown':
            unknown.append(folder.name)
            continue
        
        # Skip if not in target types
        if matra_type not in target_types:
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
    print("\n📊 Dataset statistics (available):")
    for matra_type in ['top_matra', 'side_matra', 'bottom_matra']:
        count = len(images_by_matra[matra_type])
        included = "✅ INCLUDED" if matra_type in target_types else "⏭️  SKIPPED"
        print(f"  {matra_type}: {count} images {included}")
    
    print(f"\n  Skipped (no matra): {len(skipped)} folders")
    if unknown:
        print(f"  Unknown matra type: {len(unknown)} folders")
        print(f"    Examples: {unknown[:5]}")
    
    # Sample if requested
    if sample_size:
        print(f"\n🎲 Sampling {sample_size} images per selected matra type...")
        random.seed(42)
        for matra_type in target_types:
            available = len(images_by_matra[matra_type])
            if available > sample_size:
                images_by_matra[matra_type] = random.sample(
                    images_by_matra[matra_type], sample_size
                )
                print(f"  ✓ {matra_type}: Sampled {sample_size} from {available}")
            else:
                print(f"  ⚠️  {matra_type}: Only {available} available (requested {sample_size})")
    
    # Remove non-target types
    for matra_type in list(images_by_matra.keys()):
        if matra_type not in target_types:
            images_by_matra[matra_type] = []
    
    # Copy images and create metadata
    metadata = []
    total_copied = 0
    
    print("\n📋 Copying images...")
    for matra_type, images in images_by_matra.items():
        if not images:
            continue
            
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
    with open(out_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Count by type
    counts = {
        'top_matra': len([m for m in metadata if m['matra_type'] == 'top_matra']),
        'side_matra': len([m for m in metadata if m['matra_type'] == 'side_matra']),
        'bottom_matra': len([m for m in metadata if m['matra_type'] == 'bottom_matra'])
    }
    
    # Create annotation guide
    guide_content = f"""# Modi Matra Annotation Guide

## Dataset Information
Total images prepared: {total_copied}
- Top matras: {counts['top_matra']}
- Side matras: {counts['side_matra']}
- Bottom matras: {counts['bottom_matra']}

Target matra types: {', '.join(target_types)}
Skipped folders (no matras): {len(skipped)}

## Classes (YOLO Format)
0 - top_matra (appears above base character)
1 - side_matra (appears to the right/side of base)
2 - bottom_matra (appears below base character)

## ⚠️ IMPORTANT ANNOTATION RULES

### Rule 1: Annotate ALL Matras You See! ✅

**Even if you're focusing on side matras, if an image has BOTH side + top matras:**
- Draw TWO boxes (one for each matra)
- Label each correctly based on position

**Example - KAO character (compound vowel):**
```
Image shows: Base + top mark (o component) + side mark (aa)

Your annotation:
✅ Box 1: Around top mark → Label: top_matra (class 0)
✅ Box 2: Around side mark → Label: side_matra (class 1)

❌ WRONG: Only annotating the side matra
```

### Rule 2: Label by POSITION, Not Folder Name

The folder name is just a hint. Trust your eyes:
- Mark ABOVE base → top_matra (0)
- Mark on RIGHT side → side_matra (1)  
- Mark BELOW base → bottom_matra (2)

### Rule 3: Keep Boxes Small (10-25% of image width)

✅ CORRECT: Tight box around matra only
❌ WRONG: Box covering whole character
❌ WRONG: Box including base consonant

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

## Annotation Workflow

1. **Look at the image** - Identify ALL matras present
2. **Press W** - Start drawing first box
3. **Draw tight box** around first matra (10-25% width)
4. **Select class** based on POSITION (0=top, 1=side, 2=bottom)
5. **If multiple matras exist:**
   - Press W again
   - Draw second box around second matra
   - Select its class
6. **Press D** - Next image

## Common Scenarios

### Scenario A: Single Matra Image ✅
```
Image: KAA_image.png
Contains: Base + side matra only

Action: Draw 1 box around side matra
Label: side_matra (class 1)
```

### Scenario B: Compound Matra Image ✅✅
```
Image: KAO_image.png  
Contains: Base + top mark + side mark

Action: Draw 2 boxes
Box 1: Around top mark → top_matra (0)
Box 2: Around side mark → side_matra (1)

⚠️ Don't skip the top matra just because 
   this is from a "side matra" folder!
```

### Scenario C: Triple Matra Image ✅✅✅
```
Image: Complex compound character
Contains: Top + side + bottom marks

Action: Draw 3 boxes, label each correctly
```

## Matra Position Reference

**Top Matras (Class 0):**
- i (ि) - Short vertical stroke above
- e (े) - Horizontal line above  
- ai (ै) - Two marks above
- nm/am (ं) - Dot/circle above

**Side Matras (Class 1):**
- aa (ा) - Vertical line to the right
- o (ो) - Curved mark on right
- au (ौ) - Additional marks on right
- ahaa (ः) - Two dots on right

**Bottom Matras (Class 2):**
- u (ु) - Small mark below
- uu (ू) - Longer mark below

## Quality Checks

Before finishing, verify:
- [ ] ALL matras annotated (even if not the "target" type)
- [ ] Boxes are tight (10-25% width, around matras only)
- [ ] Class labels correct (0=top, 1=side, 2=bottom based on position)
- [ ] No matras missed (especially in compound characters)
- [ ] Consistent box sizing for similar matras

## After Annotation

Once you've annotated all images:

1. Run the split creation script:
   ```
   python create_splits.py --dataset {out_dir.absolute()}
   ```

2. Train the model:
   ```
   python train_modi_matra.py --data {out_dir.absolute()}/modi_matra.yaml --epochs 150 --batch 16
   ```

## Keyboard Shortcuts in LabelImg

- W - Create bounding box
- D - Next image
- A - Previous image  
- Del - Delete selected box
- Ctrl+S - Save
- Ctrl+D - Duplicate box

## Tips for Success

✓ **Annotate ALL matras** - Even if focusing on one type
✓ **Tight boxes** - 10-25% of image width maximum
✓ **Trust position** - Not folder name
✓ **Be consistent** - Similar matras = similar boxes
✓ **Take breaks** - Every 20 images, check your quality

## Need Help?

- Check metadata.json for expected primary matra type
- Your model handles multiple matras per image automatically
- When in doubt, annotate everything you see!
"""
    
    with open(out_dir / 'ANNOTATION_GUIDE.md', 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    # Create classes.txt for LabelImg
    with open(out_dir / 'classes.txt', 'w', encoding='utf-8') as f:
        f.write("top_matra\nside_matra\nbottom_matra\n")
    
    # Create predefined_classes.txt (LabelImg looks for this)
    with open(labels_dir / 'classes.txt', 'w', encoding='utf-8') as f:
        f.write("top_matra\nside_matra\nbottom_matra\n")
    
    print(f"\n{'='*70}")
    print(f"✅ Dataset prepared successfully!")
    print(f"{'='*70}")
    print(f"\n📍 Location: {out_dir.absolute()}")
    print(f"📊 Total images: {total_copied}")
    print(f"\n📈 Distribution:")
    for matra_type, count in counts.items():
        if count > 0:
            print(f"  • {matra_type}: {count} images")
    
    print(f"\n📋 NEXT STEPS:")
    print(f"\n1. Read the annotation guide:")
    print(f"   cat {out_dir}/ANNOTATION_GUIDE.md")
    print(f"\n2. Install LabelImg (if not already installed):")
    print(f"   pip install labelImg")
    print(f"\n3. Start annotating:")
    print(f"   labelImg {images_dir.absolute()} {labels_dir.absolute()}")
    print(f"\n⚠️  REMEMBER:")
    print(f"   • Annotate ALL matras you see (not just the target type!)")
    print(f"   • Keep boxes small (10-25% width)")
    print(f"   • Label by position, not folder name")
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
  # Balanced sample (all types, 100 each)
  python prepare_modi_dataset.py --data_root ./Dataset_Modi --out datasets/modi_sample --sample 100
  
  # ONLY side matras (boost weak class)
  python prepare_modi_dataset.py --data_root ./Dataset_Modi --out datasets/modi_side_only --sample 100 --matra_types side
  
  # Side + bottom matras only
  python prepare_modi_dataset.py --data_root ./Dataset_Modi --out datasets/modi_side_bottom --sample 50 --matra_types side,bottom
  
  # Full dataset (all available)
  python prepare_modi_dataset.py --data_root ./Dataset_Modi --out datasets/modi_full
        """
    )
    parser.add_argument('--data_root', required=True, help='Root directory with character folders')
    parser.add_argument('--out', default='datasets/modi_matra', help='Output directory')
    parser.add_argument('--sample', type=int, help='Sample N images per selected matra type')
    parser.add_argument('--matra_types', type=str, help='Comma-separated matra types to include (e.g., "side,bottom" or just "side"). Options: top, side, bottom. Default: all types')
    parser.add_argument('--train_split', type=float, default=0.85, help='Train split ratio (unused for now)')
    args = parser.parse_args()
    
    if not Path(args.data_root).exists():
        print(f"ERROR: Data root not found: {args.data_root}")
        return
    
    # Parse matra types
    target_types = None
    if args.matra_types:
        target_types = [t.strip() for t in args.matra_types.split(',')]
    
    prepare_dataset(args.data_root, args.out, args.train_split, args.sample, target_types)

if __name__ == '__main__':
    main()