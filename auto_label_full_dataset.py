#!/usr/bin/env python3
"""
auto_label_full_dataset.py
Automatically label ALL remaining 6,700 images using trained model.
Separates by confidence level for different workflows.

Usage:
    python auto_label_full_dataset.py \
        --model runs/modi_matra/train2/weights/best.pt \
        --data_root "/Users/applemaair/Downloads/Dataset_Modi/Dataset_Modi" \
        --output datasets/modi_auto_labeled \
        --already_used datasets/modi_300_final/metadata.json
"""
import argparse
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import shutil
import json
import cv2

def load_already_used_images(metadata_path):
    """Load list of images already used in training."""
    if not metadata_path or not Path(metadata_path).exists():
        return set()
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    used_images = set()
    if 'images' in metadata:
        for img_info in metadata['images']:
            used_images.add(Path(img_info['path']).name)
    
    return used_images

def auto_label_dataset(model_path, data_root, output_dir, already_used_path=None):
    """Auto-label all images with confidence-based separation."""
    
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    
    # Create output structure
    for conf_level in ['high', 'medium', 'low', 'none']:
        (output_dir / 'images' / conf_level).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / conf_level).mkdir(parents=True, exist_ok=True)
    
    print("Loading trained model...")
    model = YOLO(model_path)
    
    # Get already used images
    print("Loading list of already-used images...")
    used_images = load_already_used_images(already_used_path)
    print(f"  Found {len(used_images)} images already in training set")
    
    # Find all images
    print("Scanning for images...")
    all_images = []
    for pattern in ['*.png', '*.jpg', '*.jpeg']:
        all_images.extend(data_root.rglob(pattern))
    
    # Filter out already used
    new_images = [img for img in all_images if img.name not in used_images]
    
    print(f"\nDataset Statistics:")
    print(f"  Total images found: {len(all_images)}")
    print(f"  Already used (training): {len(used_images)}")
    print(f"  New images to label: {len(new_images)}")
    
    # Process images
    print(f"\nProcessing {len(new_images)} images...")
    
    stats = {
        'high': 0,    # conf > 0.8 - auto-label directly
        'medium': 0,  # 0.5 < conf < 0.8 - quick review needed
        'low': 0,     # 0.25 < conf < 0.5 - manual annotation needed
        'none': 0     # conf < 0.25 - no detection
    }
    
    for img_path in tqdm(new_images, desc="Auto-labeling"):
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
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
                else:
                    conf_level = 'low'
            
            # Copy image
            dst_img = output_dir / 'images' / conf_level / img_path.name
            shutil.copy(img_path, dst_img)
            
            # Save labels (if any detections)
            if conf_level != 'none':
                dst_label = output_dir / 'labels' / conf_level / (img_path.stem + '.txt')
                with open(dst_label, 'w') as f:
                    boxes = results[0].boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        xywhn = box.xywhn[0].cpu().numpy()
                        f.write(f"{cls} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n")
            
            stats[conf_level] += 1
            
        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
            continue
    
    # Print results
    print("\n" + "="*70)
    print("AUTO-LABELING COMPLETE!")
    print("="*70)
    print(f"\nResults by confidence level:")
    print(f"  HIGH (>0.8):    {stats['high']:>5} images  ✅ Use directly (no review)")
    print(f"  MEDIUM (0.5-0.8): {stats['medium']:>5} images  ⚠️  Quick review (5-10 sec/image)")
    print(f"  LOW (0.25-0.5):   {stats['low']:>5} images  ❌ Manual annotation needed")
    print(f"  NONE (<0.25):     {stats['none']:>5} images  ❌ Manual annotation needed")
    print(f"\n  TOTAL PROCESSED: {sum(stats.values())} images")
    
    # Calculate time savings
    high_saved_time = stats['high'] * 60  # 60 sec per manual annotation
    medium_time = stats['medium'] * 10    # 10 sec for quick review
    low_time = (stats['low'] + stats['none']) * 60  # Full annotation
    
    print(f"\nTime Estimation:")
    print(f"  If fully manual: {sum(stats.values()) * 60 / 3600:.1f} hours")
    print(f"  With auto-labeling: {(medium_time + low_time) / 3600:.1f} hours")
    print(f"  TIME SAVED: {high_saved_time / 3600:.1f} hours! 🎉")
    
    print(f"\n{'='*70}")
    print("NEXT STEPS:")
    print("="*70)
    print("1. HIGH confidence images:")
    print(f"   → Use directly for training (already at {output_dir}/labels/high/)")
    print("\n2. MEDIUM confidence images:")
    print(f"   → Quick review needed:")
    print(f"      labelme {output_dir}/images/medium \\")
    print(f"          --labels labels.txt \\")
    print(f"          --output {output_dir}/labels_json/medium")
    print("\n3. LOW/NONE confidence images:")
    print(f"   → Full manual annotation:")
    print(f"      labelme {output_dir}/images/low \\")
    print(f"          --labels labels.txt \\")
    print(f"          --output {output_dir}/labels_json/low")
    print("\n4. Combine all and retrain:")
    print("   → Merge high + reviewed medium + annotated low")
    print("   → Total dataset: 217 + new images")
    print("   → Expected performance boost: +3-7% mAP")
    print("="*70)
    
    # Save statistics
    stats_file = output_dir / 'auto_label_stats.json'
    with open(stats_file, 'w') as f:
        json.dump({
            'total_processed': sum(stats.values()),
            'confidence_distribution': stats,
            'time_saved_hours': high_saved_time / 3600,
            'remaining_work_hours': (medium_time + low_time) / 3600
        }, f, indent=2)
    
    print(f"\nStatistics saved to: {stats_file}")

def main():
    parser = argparse.ArgumentParser(description='Auto-label full dataset with confidence separation')
    parser.add_argument('--model', required=True, help='Path to trained model (.pt)')
    parser.add_argument('--data_root', required=True, help='Root directory with all 7K images')
    parser.add_argument('--output', required=True, help='Output directory for auto-labeled data')
    parser.add_argument('--already_used', help='Path to metadata.json of already-used images')
    
    args = parser.parse_args()
    
    auto_label_dataset(args.model, args.data_root, args.output, args.already_used)

if __name__ == '__main__':
    main()
