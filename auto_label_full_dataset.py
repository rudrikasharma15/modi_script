#!/usr/bin/env python3
"""
Auto-label remaining unlabeled images using trained model.
Excludes images already used in training/validation.
"""
import argparse
import json
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import shutil

def load_used_images(dataset_dir):
    """Load list of images already used in training/validation."""
    used_images = set()
    dataset_path = Path(dataset_dir)
    
    # Check for images in train/val folders
    for split in ['train', 'val']:
        img_dir = dataset_path / 'images' / split
        if img_dir.exists():
            for img in img_dir.iterdir():
                if img.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    used_images.add(img.name)
    
    # Also check labels directory
    for split in ['train', 'val']:
        label_dir = dataset_path / 'labels' / split
        if label_dir.exists():
            for label in label_dir.iterdir():
                if label.suffix == '.txt':
                    # Add corresponding image names with all possible extensions
                    stem = label.stem
                    used_images.add(f"{stem}.png")
                    used_images.add(f"{stem}.jpg")
                    used_images.add(f"{stem}.jpeg")
    
    print(f"✓ Found {len(used_images)} already-labeled images to exclude")
    return used_images

def auto_label_dataset(model_path, data_root, output_dir, already_used_dir):
    """Auto-label remaining unlabeled images."""
    
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    
    # Load already-used images
    used_images = load_used_images(already_used_dir)
    
    # Create output directories
    for conf_level in ['high', 'medium', 'low', 'none']:
        (output_dir / 'images' / conf_level).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / conf_level).mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\n📦 Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Find all images recursively
    print(f"\n🔍 Scanning {data_root} for images...")
    all_images = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        all_images.extend(list(data_root.rglob(ext)))
    
    print(f"✓ Found {len(all_images)} total images")
    
    # Filter out already-used images
    new_images = [img for img in all_images if img.name not in used_images]
    
    print(f"✓ Excluding {len(all_images) - len(new_images)} already-labeled images")
    print(f"✓ Processing {len(new_images)} NEW images\n")
    
    if len(new_images) == 0:
        print("❌ No new images to process!")
        return
    
    # Counters
    counts = {'high': 0, 'medium': 0, 'low': 0, 'none': 0}
    
    print("🚀 Auto-labeling images...")
    for img_path in tqdm(new_images):
        try:
            # Run inference
            results = model.predict(img_path, conf=0.25, verbose=False)
            
            # Determine confidence level
            if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
                conf_level = 'none'
                max_conf = 0
            else:
                boxes = results[0].boxes
                confidences = [float(box.conf[0]) for box in boxes]
                max_conf = max(confidences)
                
                if max_conf >= 0.8:
                    conf_level = 'high'
                elif max_conf >= 0.5:
                    conf_level = 'medium'
                elif max_conf >= 0.25:
                    conf_level = 'low'
                else:
                    conf_level = 'none'
            
            # Copy image
            dst_img = output_dir / 'images' / conf_level / img_path.name
            shutil.copy2(img_path, dst_img)
            
            # Save labels (if any detections)
            if conf_level != 'none' and len(results[0].boxes) > 0:
                dst_label = output_dir / 'labels' / conf_level / (img_path.stem + '.txt')
                with open(dst_label, 'w') as f:
                    for box in results[0].boxes:
                        cls = int(box.cls[0])
                        xywhn = box.xywhn[0].cpu().numpy()
                        conf = float(box.conf[0])
                        f.write(f"{cls} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n")
            
            counts[conf_level] += 1
            
        except Exception as e:
            print(f"\n⚠️  Error processing {img_path.name}: {e}")
            continue
    
    # Calculate time savings
    total_processed = sum(counts.values())
    manual_time_hours = total_processed * 30 / 3600  # 30 sec per image
    high_auto_time = counts['high'] * 0  # No time needed
    medium_review_time = counts['medium'] * 10 / 3600  # 10 sec per image
    low_manual_time = counts['low'] * 30 / 3600  # Full annotation
    none_manual_time = counts['none'] * 30 / 3600  # Full annotation
    
    actual_time_hours = high_auto_time + medium_review_time + low_manual_time + none_manual_time
    time_saved = manual_time_hours - actual_time_hours
    
    # Save metadata
    metadata = {
        'model_used': str(model_path),
        'images_processed': total_processed,
        'excluded_already_labeled': len(all_images) - len(new_images),
        'confidence_distribution': counts,
        'time_analysis': {
            'if_all_manual_hours': round(manual_time_hours, 1),
            'with_semi_supervised_hours': round(actual_time_hours, 1),
            'time_saved_hours': round(time_saved, 1),
            'efficiency_gain_percent': round((time_saved / manual_time_hours * 100), 1) if manual_time_hours > 0 else 0
        }
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print results
    print("\n" + "="*60)
    print("🎉 AUTO-LABELING COMPLETE!")
    print("="*60)
    print(f"\n📊 Results by confidence level:")
    print(f"  ✅ HIGH (>0.8):     {counts['high']:,} images  → Use directly")
    print(f"  ⚠️  MEDIUM (0.5-0.8): {counts['medium']:,} images  → Quick review")
    print(f"  ❌ LOW (0.25-0.5):   {counts['low']:,} images  → Manual work")
    print(f"  ❌ NONE (<0.25):     {counts['none']:,} images  → Manual work")
    print(f"\n📁 Total processed: {total_processed:,} NEW images")
    print(f"🚫 Excluded (already labeled): {len(all_images) - len(new_images):,} images")
    
    print(f"\n⏱️  TIME ANALYSIS:")
    print(f"  If all manual:        {manual_time_hours:.1f} hours")
    print(f"  With semi-supervised: {actual_time_hours:.1f} hours")
    print(f"  TIME SAVED:           {time_saved:.1f} hours ({metadata['time_analysis']['efficiency_gain_percent']:.0f}% reduction) 🎉")
    
    print(f"\n📝 NEXT STEPS:")
    print(f"  1. HIGH: Use directly for training ({counts['high']:,} images)")
    print(f"  2. MEDIUM: Quick review in labelme ({medium_review_time:.1f} hours)")
    print(f"  3. LOW/NONE: Manual annotation or skip for now ({low_manual_time + none_manual_time:.1f} hours)")
    
    print(f"\n💾 Output saved to: {output_dir}")
    print(f"📄 Metadata saved to: {output_dir / 'metadata.json'}")
    print("="*60 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Auto-label remaining unlabeled images')
    parser.add_argument('--model', required=True, help='Path to trained model (.pt file)')
    parser.add_argument('--data_root', required=True, help='Root directory of all Modi images')
    parser.add_argument('--output', required=True, help='Output directory for auto-labeled images')
    parser.add_argument('--already_used', required=True, help='Directory of already-labeled dataset')
    
    args = parser.parse_args()
    
    auto_label_dataset(args.model, args.data_root, args.output, args.already_used)