#!/usr/bin/env python3
"""
Visualize test set results with detailed metrics and sample predictions.
"""
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def plot_confusion_metrics(results_dir):
    """Plot detailed metrics from test results."""
    results_dir = Path(results_dir)
    
    # Check if confusion matrix exists
    confusion_path = results_dir / "confusion_matrix.png"
    results_path = results_dir / "results.png"
    
    print("\n" + "="*70)
    print("📊 TEST RESULTS VISUALIZATION")
    print("="*70)
    
    if confusion_path.exists():
        print(f"✅ Confusion Matrix: {confusion_path}")
    if results_path.exists():
        print(f"✅ Training Curves: {results_path}")
    
    # Load results CSV if available
    results_csv = results_dir / "results.csv"
    if results_csv.exists():
        print(f"✅ Detailed Metrics: {results_csv}")

def predict_and_visualize_test_samples(model_path, test_dir, output_dir, num_samples=20):
    """
    Run predictions on test set and visualize sample results.
    """
    model = YOLO(model_path)
    test_dir = Path(test_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get test images
    test_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    
    if not test_images:
        print(f"❌ No images found in {test_dir}")
        return
    
    print(f"\n📸 Found {len(test_images)} test images")
    
    # Sample images
    import random
    random.seed(42)
    sample_images = random.sample(test_images, min(num_samples, len(test_images)))
    
    # Class names and colors
    class_names = {0: 'top_matra', 1: 'side_matra', 2: 'bottom_matra'}
    colors = {
        0: (255, 0, 0),      # Red for top
        1: (0, 255, 0),      # Green for side
        2: (0, 0, 255)       # Blue for bottom
    }
    
    # Statistics
    stats = defaultdict(int)
    all_confidences = defaultdict(list)
    
    print(f"\n🔍 Running predictions on {len(sample_images)} sample images...")
    
    for i, img_path in enumerate(sample_images, 1):
        # Run prediction
        results = model.predict(img_path, conf=0.25, verbose=False)[0]
        
        # Load image
        img = cv2.imread(str(img_path))
        img_display = img.copy()
        
        # Draw predictions
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Statistics
            stats[class_names[cls]] += 1
            all_confidences[class_names[cls]].append(conf)
            
            # Draw box
            color = colors[cls]
            cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"{class_names[cls]}: {conf:.2f}"
            cv2.putText(img_display, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save
        output_path = output_dir / f"test_sample_{i:03d}_{img_path.name}"
        cv2.imwrite(str(output_path), img_display)
        
        if i % 5 == 0:
            print(f"  Processed {i}/{len(sample_images)}...")
    
    print(f"\n✅ Saved visualizations to: {output_dir}")
    
    # Print statistics
    print("\n" + "="*70)
    print("📊 DETECTION STATISTICS (Sample)")
    print("="*70)
    print(f"Total detections: {sum(stats.values())}")
    print(f"\nPer-class detections:")
    for cls_name in ['top_matra', 'side_matra', 'bottom_matra']:
        count = stats[cls_name]
        avg_conf = np.mean(all_confidences[cls_name]) if all_confidences[cls_name] else 0
        print(f"  {cls_name:15s}: {count:3d} detections (avg conf: {avg_conf:.3f})")
    print("="*70)

def generate_summary_report(model_path, test_yaml, output_dir):
    """Generate comprehensive test report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("📋 GENERATING COMPREHENSIVE TEST REPORT")
    print("="*70)
    
    # Run validation on test set
    model = YOLO(model_path)
    results = model.val(data=test_yaml, split='test', save_json=True, plots=True)
    
    # Extract metrics
    report = {
        'Overall': {
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'Precision': float(results.box.mp),
            'Recall': float(results.box.mr)
        },
        'Per-Class': {}
    }
    
    class_names = ['top_matra', 'side_matra', 'bottom_matra']
    for i, cls_name in enumerate(class_names):
        report['Per-Class'][cls_name] = {
            'mAP50': float(results.box.maps[i]),
            'Precision': float(results.box.p[i]),
            'Recall': float(results.box.r[i])
        }
    
    # Save report
    report_path = output_dir / 'test_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Report saved to: {report_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("📊 TEST SET PERFORMANCE SUMMARY")
    print("="*70)
    print(f"\nOverall Metrics:")
    print(f"  mAP50:     {report['Overall']['mAP50']:.3f} ({report['Overall']['mAP50']*100:.1f}%)")
    print(f"  mAP50-95:  {report['Overall']['mAP50-95']:.3f} ({report['Overall']['mAP50-95']*100:.1f}%)")
    print(f"  Precision: {report['Overall']['Precision']:.3f} ({report['Overall']['Precision']*100:.1f}%)")
    print(f"  Recall:    {report['Overall']['Recall']:.3f} ({report['Overall']['Recall']*100:.1f}%)")
    
    print(f"\nPer-Class Performance:")
    print(f"{'Class':<15} {'mAP50':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 45)
    for cls_name, metrics in report['Per-Class'].items():
        print(f"{cls_name:<15} {metrics['mAP50']:>7.3f}  {metrics['Precision']:>9.3f}  {metrics['Recall']:>7.3f}")
    print("="*70)
    
    return report

def main():
    parser = argparse.ArgumentParser(description='Visualize test set results')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--test_images', required=True, help='Path to test images directory')
    parser.add_argument('--test_yaml', required=True, help='Path to dataset YAML file')
    parser.add_argument('--output', default='test_visualizations', help='Output directory')
    parser.add_argument('--samples', type=int, default=20, help='Number of sample images to visualize')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("🎯 MODI MATRA TEST SET EVALUATION")
    print("="*70)
    print(f"Model:       {args.model}")
    print(f"Test Images: {args.test_images}")
    print(f"Output:      {args.output}")
    print("="*70)
    
    # 1. Generate comprehensive report
    print("\n[1/3] Generating test metrics...")
    report = generate_summary_report(args.model, args.test_yaml, output_dir)
    
    # 2. Visualize sample predictions
    print("\n[2/3] Visualizing sample predictions...")
    predict_and_visualize_test_samples(
        args.model, 
        args.test_images, 
        output_dir / 'sample_predictions',
        num_samples=args.samples
    )
    
    # 3. Check for plots
    print("\n[3/3] Checking for result plots...")
    # Results should be in runs/detect/val* after model.val()
    
    print("\n" + "="*70)
    print("✅ TEST EVALUATION COMPLETE!")
    print("="*70)
    print(f"\n📁 All results saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  • test_report.json - Detailed metrics")
    print(f"  • sample_predictions/ - Visualized predictions")
    print(f"  • Check runs/detect/val*/ for confusion matrix & plots")
    print("="*70)

if __name__ == '__main__':
    main()