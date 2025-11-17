#!/usr/bin/env python3
"""
Check confidence scores after manual review.
Compare model predictions with your corrected labels.
"""
import argparse
import json
from pathlib import Path
from ultralytics import YOLO
from collections import Counter
import numpy as np

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    if xi_max <= xi_min or yi_max <= yi_min:
        return 0.0
    
    intersection = (xi_max - xi_min) * (yi_max - yi_min)
    
    # Union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def yolo_to_corners(x_center, y_center, width, height, img_width, img_height):
    """Convert YOLO format to corner coordinates."""
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    x_min = x_center_px - width_px / 2
    y_min = y_center_px - height_px / 2
    x_max = x_center_px + width_px / 2
    y_max = y_center_px + height_px / 2
    
    return [x_min, y_min, x_max, y_max]

def check_reviewed_labels(model_path, json_dir, output_file):
    """Re-run model on reviewed images and compare results."""
    
    json_dir = Path(json_dir)
    model = YOLO(model_path)
    
    print(f"🔍 Loading model: {model_path}")
    print(f"📂 Checking reviewed labels in: {json_dir}\n")
    
    # Find all JSON files
    json_files = list(json_dir.glob('*.json'))
    print(f"Found {len(json_files)} reviewed images\n")
    
    results = {
        'improved': [],      # Better match after review
        'same': [],          # No change
        'worse': [],         # Worse after review (rare)
        'corrections': [],   # Manual corrections made
    }
    
    stats = Counter()
    confidence_changes = []
    
    for json_file in json_files:
        # Load reviewed labels
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        img_path = json_dir / data['imagePath']
        if not img_path.exists():
            stats['missing_image'] += 1
            continue
        
        img_width = data['imageWidth']
        img_height = data['imageHeight']
        
        # Get ground truth (your reviewed labels)
        gt_boxes = []
        for shape in data['shapes']:
            if shape['shape_type'] == 'rectangle':
                points = shape['points']
                x_min = min(points[0][0], points[1][0])
                y_min = min(points[0][1], points[1][1])
                x_max = max(points[0][0], points[1][0])
                y_max = max(points[0][1], points[1][1])
                
                label = shape['label']
                class_id = {'top_matra': 0, 'side_matra': 1, 'bottom_matra': 2}.get(label, -1)
                
                gt_boxes.append({
                    'class': class_id,
                    'box': [x_min, y_min, x_max, y_max]
                })
        
        # Run model prediction
        pred_results = model.predict(img_path, conf=0.25, verbose=False)
        
        if len(pred_results) == 0 or pred_results[0].boxes is None:
            if len(gt_boxes) > 0:
                results['corrections'].append(json_file.name)
                stats['added_missing'] += 1
            continue
        
        # Get predictions
        pred_boxes = []
        for box in pred_results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xywhn = box.xywhn[0].cpu().numpy()
            
            box_coords = yolo_to_corners(
                xywhn[0], xywhn[1], xywhn[2], xywhn[3],
                img_width, img_height
            )
            
            pred_boxes.append({
                'class': cls,
                'confidence': conf,
                'box': box_coords
            })
        
        # Compare predictions with ground truth
        best_ious = []
        for gt_box in gt_boxes:
            best_iou = 0
            best_conf = 0
            for pred_box in pred_boxes:
                if pred_box['class'] == gt_box['class']:
                    iou = calculate_iou(gt_box['box'], pred_box['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_conf = pred_box['confidence']
            
            if best_iou > 0.5:  # Match found
                best_ious.append(best_iou)
                confidence_changes.append(best_conf)
                
                if best_conf >= 0.8:
                    results['improved'].append(json_file.name)
                elif best_conf >= 0.5:
                    results['same'].append(json_file.name)
                else:
                    results['worse'].append(json_file.name)
            else:
                # Manual correction - no good match with prediction
                results['corrections'].append(json_file.name)
                stats['manual_corrections'] += 1
        
        stats['processed'] += 1
    
    # Calculate statistics
    avg_confidence = np.mean(confidence_changes) if confidence_changes else 0
    
    # Save detailed results
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_reviewed': len(json_files),
                'processed': stats['processed'],
                'average_confidence': round(float(avg_confidence), 3),
                'improved_to_high': len(results['improved']),
                'stayed_medium': len(results['same']),
                'manual_corrections': len(results['corrections']),
            },
            'details': results
        }, f, indent=2)
    
    # Print summary
    print("="*60)
    print("📊 REVIEW QUALITY ANALYSIS")
    print("="*60)
    print(f"\n✅ Processed: {stats['processed']} images")
    print(f"📈 Average confidence on reviewed: {avg_confidence:.3f}")
    print(f"\n🎯 Confidence Distribution:")
    print(f"  High (>0.8):     {len(results['improved'])} images")
    print(f"  Medium (0.5-0.8): {len(results['same'])} images")
    print(f"  Manual fixes:     {len(results['corrections'])} images")
    
    if len(results['corrections']) > 0:
        correction_pct = len(results['corrections']) / stats['processed'] * 100
        print(f"\n✏️  Manual correction rate: {correction_pct:.1f}%")
        print(f"   (These are images where you fixed significant errors)")
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    print("="*60)
    
    # Recommendations
    print("\n📝 RECOMMENDATIONS:")
    
    if avg_confidence >= 0.8:
        print("  ✅ Excellent! Most reviewed labels now have high confidence.")
        print("  ✅ Ready to merge and retrain.")
    elif avg_confidence >= 0.6:
        print("  ✅ Good! Your reviews improved label quality.")
        print("  ✅ Can proceed to merge and retrain.")
    else:
        print("  ⚠️  Some labels still have lower confidence.")
        print("  💡 Consider reviewing more images or checking complex cases.")
    
    if len(results['corrections']) > stats['processed'] * 0.3:
        print(f"\n  🎯 High manual correction rate ({len(results['corrections'])} images)")
        print("  💡 This is good - you're catching and fixing model errors!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check confidence of reviewed labels')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--json_dir', required=True, help='Directory with reviewed labelme JSONs')
    parser.add_argument('--output', default='review_analysis.json', help='Output JSON file')
    
    args = parser.parse_args()
    
    check_reviewed_labels(args.model, args.json_dir, args.output)
    