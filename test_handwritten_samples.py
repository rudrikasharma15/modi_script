#!/usr/bin/env python3
"""
Test model on handwritten samples and generate detailed report.
"""
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import json
from collections import defaultdict

def test_handwritten_samples(model_path, samples_dir, output_dir, conf_threshold=0.25):
    """Test model on handwritten samples."""
    
    model = YOLO(model_path)
    samples_path = Path(samples_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(list(samples_path.glob(ext)))
    
    if not image_files:
        print(f"❌ No images found in {samples_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"🧪 TESTING ON HANDWRITTEN SAMPLES")
    print(f"{'='*70}\n")
    print(f"Model: {model_path}")
    print(f"Samples: {samples_dir}")
    print(f"Found {len(image_files)} images\n")
    
    # Statistics
    stats = {
        'total_images': len(image_files),
        'successful_detections': 0,
        'failed_detections': 0,
        'detections_by_class': defaultdict(int),
        'detections_by_size': defaultdict(int),
        'confidence_scores': defaultdict(list),
        'results': []
    }
    
    class_names = {0: 'top_matra', 1: 'side_matra', 2: 'bottom_matra'}
    colors = {
        'top_matra': (0, 0, 255),      # Red
        'side_matra': (0, 255, 0),     # Green
        'bottom_matra': (255, 0, 0)    # Blue
    }
    
    # Process each image
    for img_file in sorted(image_files):
        print(f"Processing: {img_file.name}")
        
        # Extract expected info from filename
        filename = img_file.stem.lower()
        expected_size = 'unknown'
        if 'small' in filename:
            expected_size = 'small'
        elif 'medium' in filename:
            expected_size = 'medium'
        elif 'large' in filename:
            expected_size = 'large'
        
        # Run inference
        results = model(img_file, conf=conf_threshold, verbose=False)
        
        if len(results) == 0:
            stats['failed_detections'] += 1
            stats['results'].append({
                'image': img_file.name,
                'size': expected_size,
                'detected': False,
                'detections': []
            })
            continue
        
        result = results[0]
        boxes = result.boxes
        
        # Load image for visualization
        img = cv2.imread(str(img_file))
        
        detections = []
        
        if len(boxes) > 0:
            stats['successful_detections'] += 1
            
            for box in boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                xyxy = box.xyxy[0].cpu().numpy()
                
                class_name = class_names[cls_id]
                
                # Update statistics
                stats['detections_by_class'][class_name] += 1
                stats['detections_by_size'][expected_size] += 1
                stats['confidence_scores'][class_name].append(conf)
                
                # Draw on image
                x1, y1, x2, y2 = map(int, xyxy)
                color = colors.get(class_name, (255, 255, 255))
                
                # Draw box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                
                # Draw label with larger font
                label = f"{class_name}: {conf:.2f}"
                font_scale = 1.2
                thickness = 2
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Background for label
                cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, (255, 255, 255), thickness)
                
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
        else:
            stats['failed_detections'] += 1
        
        # Save annotated image
        output_file = output_path / f"detected_{img_file.name}"
        cv2.imwrite(str(output_file), img)
        
        stats['results'].append({
            'image': img_file.name,
            'size': expected_size,
            'detected': len(detections) > 0,
            'detections': detections
        })
    
    # Calculate average confidences
    avg_confidences = {}
    for class_name, scores in stats['confidence_scores'].items():
        if scores:
            avg_confidences[class_name] = sum(scores) / len(scores)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"📊 RESULTS SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"Total images tested: {stats['total_images']}")
    print(f"Successful detections: {stats['successful_detections']} ({stats['successful_detections']/stats['total_images']*100:.1f}%)")
    print(f"Failed detections: {stats['failed_detections']} ({stats['failed_detections']/stats['total_images']*100:.1f}%)")
    
    print(f"\n📈 Detections by Class:")
    for class_name, count in sorted(stats['detections_by_class'].items()):
        avg_conf = avg_confidences.get(class_name, 0)
        print(f"  {class_name:<15}: {count:>3} detections (avg confidence: {avg_conf:.3f})")
    
    print(f"\n📏 Detections by Size:")
    for size, count in sorted(stats['detections_by_size'].items()):
        print(f"  {size:<10}: {count:>3} detections")
    
    # Success rate analysis
    print(f"\n✅ Success Rate Analysis:")
    success_rate = stats['successful_detections'] / stats['total_images'] * 100
    
    if success_rate >= 85:
        verdict = "EXCELLENT"
        emoji = "🎉"
    elif success_rate >= 70:
        verdict = "GOOD"
        emoji = "✅"
    elif success_rate >= 50:
        verdict = "FAIR"
        emoji = "⚠️"
    else:
        verdict = "POOR"
        emoji = "❌"
    
    print(f"  {emoji} {verdict}: {success_rate:.1f}% success rate")
    
    if success_rate >= 70:
        print(f"  → Model generalizes well to handwritten samples!")
    else:
        print(f"  → Model may need more training on diverse handwriting styles")
    
    # Save detailed report
    report = {
        'model': str(model_path),
        'samples_directory': str(samples_dir),
        'total_images': stats['total_images'],
        'successful_detections': stats['successful_detections'],
        'failed_detections': stats['failed_detections'],
        'success_rate': success_rate,
        'detections_by_class': dict(stats['detections_by_class']),
        'detections_by_size': dict(stats['detections_by_size']),
        'average_confidences': avg_confidences,
        'detailed_results': stats['results']
    }
    
    report_file = output_path / 'handwritten_test_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create markdown report
    md_report = f"""# Handwritten Sample Testing Report

## Overview
- **Model**: {model_path}
- **Test Images**: {stats['total_images']}
- **Success Rate**: {success_rate:.1f}%

## Results Summary

### Detection Statistics
- ✅ Successful: {stats['successful_detections']} ({stats['successful_detections']/stats['total_images']*100:.1f}%)
- ❌ Failed: {stats['failed_detections']} ({stats['failed_detections']/stats['total_images']*100:.1f}%)

### By Matra Class
| Class | Detections | Avg Confidence |
|-------|-----------|----------------|
"""
    
    for class_name in sorted(stats['detections_by_class'].keys()):
        count = stats['detections_by_class'][class_name]
        avg_conf = avg_confidences.get(class_name, 0)
        md_report += f"| {class_name} | {count} | {avg_conf:.3f} |\n"
    
    md_report += f"""
### By Character Size
| Size | Detections |
|------|-----------|
"""
    
    for size in sorted(stats['detections_by_size'].keys()):
        count = stats['detections_by_size'][size]
        md_report += f"| {size} | {count} |\n"
    
    md_report += f"""
## Verdict
**{verdict}** - {success_rate:.1f}% success rate

## Files
- Annotated images: `{output_path}/`
- Detailed JSON report: `{report_file.name}`
"""
    
    md_file = output_path / 'REPORT.md'
    with open(md_file, 'w') as f:
        f.write(md_report)
    
    print(f"\n📄 Reports saved:")
    print(f"  - JSON: {report_file}")
    print(f"  - Markdown: {md_file}")
    print(f"  - Images: {output_path}/")
    print(f"\n{'='*70}\n")
    
    return report

def main():
    parser = argparse.ArgumentParser(description='Test model on handwritten samples')
    parser.add_argument('--model', required=True, help='Path to trained model (.pt)')
    parser.add_argument('--samples', required=True, help='Directory with handwritten sample images')
    parser.add_argument('--output', default='handwritten_test_results', help='Output directory')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    
    args = parser.parse_args()
    
    test_handwritten_samples(args.model, args.samples, args.output, args.conf)

if __name__ == '__main__':
    main()