"""
FIXED: Hierarchical YOLO Dataset - Consecutive Class IDs
"""

import os
import shutil
from pathlib import Path
import re
import json

def normalize_folder_name(folder_name):
    """Normalize folder name to extract base character."""
    folder_name = re.sub(r'^\d+\s+', '', folder_name)
    base = folder_name.split('-')[0].lower().strip()
    return base

def build_character_class_mapping(ieee_path):
    """Build comprehensive class mapping with CONSECUTIVE IDs."""
    ieee_path = Path(ieee_path)
    
    # Collect ALL folders (excluding vowel-only)
    all_folders = []
    
    for folder in sorted(ieee_path.iterdir()):
        if not folder.is_dir():
            continue
        
        # Skip vowel-only folders
        base = normalize_folder_name(folder.name)
        if base in ['a', 'aa', 'i', 'u', 'e', 'ai', 'o', 'au', 'nm', 'ahaa']:
            continue
        
        # Check if folder has images
        has_images = False
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.JPG', '*.JPEG', '*.PNG', '*.TIF']:
            if list(folder.glob(ext)):
                has_images = True
                break
        
        if has_images:
            all_folders.append(folder.name)
    
    # Assign CONSECUTIVE class IDs
    class_mapping = {}
    for idx, folder_name in enumerate(sorted(all_folders)):
        class_mapping[folder_name] = {
            'class': idx,
            'name': normalize_folder_name(folder_name)
        }
    
    # Add matra detection classes at the end
    num_char_classes = len(all_folders)
    class_mapping['_MATRA_TOP'] = {'class': num_char_classes, 'name': 'top_matra'}
    class_mapping['_MATRA_SIDE'] = {'class': num_char_classes + 1, 'name': 'side_matra'}
    class_mapping['_MATRA_BOTTOM'] = {'class': num_char_classes + 2, 'name': 'bottom_matra'}
    
    total_classes = num_char_classes + 3
    
    return class_mapping, total_classes, num_char_classes

def find_matra_labels(image_path, matra_labels_dir):
    """Find corresponding matra bounding boxes."""
    img_name = Path(image_path).stem
    matra_labels_path = Path(matra_labels_dir)
    
    label_path = matra_labels_path / f"{img_name}.txt"
    
    if label_path.exists():
        matras = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    matras.append([int(parts[0])] + list(map(float, parts[1:])))
        return matras
    
    return []

def create_hierarchical_labels(
    ieee_dataset_path,
    matra_labels_path,
    output_path,
    train_split=0.7,
    val_split=0.15
):
    """Create complete hierarchical dataset with consecutive class IDs."""
    ieee_path = Path(ieee_dataset_path)
    matra_path = Path(matra_labels_path)
    output_path = Path(output_path)
    
    print("\n" + "="*60)
    print("COMPLETE HIERARCHICAL YOLO DATASET")
    print("="*60)
    
    # Build class mapping
    print("Building class mapping with consecutive IDs...")
    class_mapping, total_classes, matra_start_class = build_character_class_mapping(ieee_path)
    
    print(f"Total classes: {total_classes}")
    print(f"  - Character classes (0-{matra_start_class-1}): {matra_start_class} classes")
    print(f"  - Matra detection ({matra_start_class}-{total_classes-1}): 3 classes")
    print("="*60 + "\n")
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    stats = {
        'total_images': 0,
        'total_character_boxes': 0,
        'total_matra_boxes': 0,
        'images_with_matras': 0,
        'images_without_matras': 0
    }
    
    all_images = []
    
    # Process all folders
    for folder in sorted(ieee_path.iterdir()):
        if not folder.is_dir():
            continue
        
        # Skip if not in mapping
        if folder.name not in class_mapping or folder.name.startswith('_'):
            continue
        
        folder_info = class_mapping[folder.name]
        char_class = folder_info['class']
        
        print(f"📁 Class {char_class:3d}: {folder.name}")
        
        # Find all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.JPG', '*.JPEG', '*.PNG', '*.TIF']
        folder_images = []
        for ext in image_extensions:
            folder_images.extend(list(folder.glob(ext)))
        
        if len(folder_images) == 0:
            continue
        
        print(f"         Found {len(folder_images)} images")
        
        # Process images
        for img_path in folder_images:
            # Find matras
            matras = find_matra_labels(img_path, matra_path)
            
            # Convert matra classes to detection classes
            matra_boxes = []
            for matra in matras:
                matra_class = matra_start_class + matra[0]
                matra_boxes.append([matra_class] + matra[1:])
            
            all_images.append({
                'image_path': img_path,
                'char_class': char_class,
                'matras': matra_boxes
            })
            
            stats['total_images'] += 1
            stats['total_character_boxes'] += 1
            stats['total_matra_boxes'] += len(matra_boxes)
            
            if matra_boxes:
                stats['images_with_matras'] += 1
            else:
                stats['images_without_matras'] += 1
    
    print(f"\n✅ Collected {len(all_images)} total images")
    
    # Split dataset
    import random
    random.seed(42)
    random.shuffle(all_images)
    
    train_end = int(len(all_images) * train_split)
    val_end = train_end + int(len(all_images) * val_split)
    
    splits = {
        'train': all_images[:train_end],
        'val': all_images[train_end:val_end],
        'test': all_images[val_end:]
    }
    
    # Write labels and copy images
    for split_name, images in splits.items():
        print(f"\n📝 Writing {split_name} set ({len(images)} images)...")
        
        for idx, item in enumerate(images):
            # Copy image
            img_ext = item['image_path'].suffix
            img_name = f"{split_name}_{idx:05d}{img_ext}"
            dst_img = output_path / split_name / 'images' / img_name
            
            try:
                shutil.copy2(item['image_path'], dst_img)
            except Exception as e:
                continue
            
            # Create label file
            label_path = output_path / split_name / 'labels' / f"{split_name}_{idx:05d}.txt"
            
            with open(label_path, 'w') as f:
                # Character box (full image)
                f.write(f"{item['char_class']} 0.5 0.5 1.0 1.0\n")
                
                # Matra boxes
                for matra in item['matras']:
                    f.write(f"{int(matra[0])} {matra[1]:.6f} {matra[2]:.6f} {matra[3]:.6f} {matra[4]:.6f}\n")
    
    # Create data.yaml with CONSECUTIVE class names
    class_names_list = [''] * total_classes
    
    for folder_name, info in class_mapping.items():
        class_id = info['class']
        class_name = info['name']
        class_names_list[class_id] = class_name
    
    yaml_content = f"""# Complete Hierarchical Modi Dataset
path: {output_path.absolute()}
train: train/images
val: val/images
test: test/images

nc: {total_classes}

names:
"""
    
    for cls_id, cls_name in enumerate(class_names_list):
        yaml_content += f"  {cls_id}: {cls_name}\n"
    
    with open(output_path / 'data.yaml', 'w') as f:
        f.write(yaml_content)
    
    # Save class mapping
    with open(output_path / 'class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    # Print statistics
    print("\n" + "="*60)
    print("DATASET CREATION COMPLETE")
    print("="*60)
    print(f"Total images: {stats['total_images']}")
    print(f"Total character classes: {matra_start_class}")
    print(f"Total matra classes: 3")
    print(f"Total classes: {total_classes}")
    print(f"\nTotal boxes:")
    print(f"  - Character boxes: {stats['total_character_boxes']}")
    print(f"  - Matra boxes: {stats['total_matra_boxes']}")
    print(f"\nImages with matras: {stats['images_with_matras']}")
    print(f"Images without matras: {stats['images_without_matras']}")
    
    if stats['total_images'] > 0:
        avg_boxes = (stats['total_character_boxes'] + stats['total_matra_boxes']) / stats['total_images']
        print(f"Average boxes per image: {avg_boxes:.2f}")
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(splits['train'])} ({len(splits['train'])/len(all_images)*100:.1f}%)")
    print(f"  Val: {len(splits['val'])} ({len(splits['val'])/len(all_images)*100:.1f}%)")
    print(f"  Test: {len(splits['test'])} ({len(splits['test'])/len(all_images)*100:.1f}%)")
    
    print(f"\n✅ Dataset ready: {output_path}")
    print(f"✅ Config: {output_path / 'data.yaml'}")
    print(f"✅ Class mapping: {output_path / 'class_mapping.json'}")
    
    return stats

if __name__ == "__main__":
    IEEE_DATASET = r"C:\Users\admin\Desktop\modi script\modi_script\Dataset_Modi"
    MATRA_LABELS = r"C:\Users\admin\Desktop\modi script\modi_script\modi_full_7k\merged\labels"
    OUTPUT_DIR = r"C:\Users\admin\Desktop\modi script\modi_script\hierarchical_dataset_full"
    
    stats = create_hierarchical_labels(
        ieee_dataset_path=IEEE_DATASET,
        matra_labels_path=MATRA_LABELS,
        output_path=OUTPUT_DIR
    )
    
    if stats['total_images'] > 0:
        print("\n🚀 TRAIN WITH:")
        print(f'yolo task=detect mode=train data="{OUTPUT_DIR}\\data.yaml" model=yolov8n.pt epochs=150 imgsz=640 batch=16')