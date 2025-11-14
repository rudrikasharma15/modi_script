#!/usr/bin/env python3
"""
predict_modi_matra.py
Run inference on Modi script images to detect matras.

Usage:
    # Single image
    python predict_modi_matra.py --model runs/modi_matra/train/weights/best.pt --source image.jpg
    
    # Folder of images
    python predict_modi_matra.py --model runs/modi_matra/train/weights/best.pt --source test_images/
    
    # With custom confidence
    python predict_modi_matra.py --model best.pt --source test/ --conf 0.35
"""
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import json

# Class names and colors
CLASS_NAMES = {
    0: 'top_matra',
    1: 'side_matra',
    2: 'bottom_matra'
}

CLASS_COLORS = {
    0: (0, 255, 0),      # top_matra - Green
    1: (255, 0, 0),      # side_matra - Blue
    2: (0, 0, 255),      # bottom_matra - Red
}

def predict_and_visualize(model_path, source, conf=0.25, save_dir='predictions', save_json=False):
    """
    Run inference and save visualizations.
    
    Args:
        model_path: Path to trained model
        source: Image file or directory
        conf: Confidence threshold
        save_dir: Output directory
        save_json: Whether to save predictions as JSON
    """
    
    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Create output directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    print(f"Running inference on: {source}")
    print(f"Confidence threshold: {conf}")
    
    results = model.predict(
        source=source,
        conf=conf,
        save=False,  # We'll handle saving ourselves
        show=False,
        verbose=True
    )
    
    # Process results
    all_predictions = []
    
    for i, result in enumerate(results):
        # Get image
        img = result.orig_img.copy()
        img_name = Path(result.path).name if hasattr(result, 'path') else f"image_{i}.jpg"
        
        # Get predictions
        boxes = result.boxes
        predictions = []
        
        if len(boxes) > 0:
            for box in boxes:
                # Get box info
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Draw box
                color = CLASS_COLORS.get(cls_id, (255, 255, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{CLASS_NAMES.get(cls_id, f'class_{cls_id}')} {conf_score:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                # Background for label
                cv2.rectangle(img, (x1, y1 - label_size[1] - 5), 
                            (x1 + label_size[0], y1), color, -1)
                
                # Label text
                cv2.putText(img, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Store prediction
                predictions.append({
                    'class': CLASS_NAMES.get(cls_id, f'class_{cls_id}'),
                    'class_id': cls_id,
                    'confidence': conf_score,
                    'bbox': [x1, y1, x2, y2]
                })
        
        # Save image
        output_path = save_dir / img_name
        cv2.imwrite(str(output_path), img)
        
        # Store all predictions
        all_predictions.append({
            'image': img_name,
            'predictions': predictions
        })
        
        # Print summary
        matra_counts = {}
        for pred in predictions:
            cls = pred['class']
            matra_counts[cls] = matra_counts.get(cls, 0) + 1
        
        print(f"  {img_name}: {len(predictions)} matras detected", end='')
        if matra_counts:
            counts_str = ', '.join([f"{k}={v}" for k, v in matra_counts.items()])
            print(f" ({counts_str})")
        else:
            print()
    
    # Save JSON if requested
    if save_json:
        json_path = save_dir / 'predictions.json'
        with open(json_path, 'w') as f:
            json.dump(all_predictions, f, indent=2)
        print(f"\nPredictions saved to: {json_path}")
    
    # Print summary
    total_images = len(results)
    total_detections = sum(len(p['predictions']) for p in all_predictions)
    
    print(f"\n{'='*60}")
    print(f"Prediction Summary:")
    print(f"{'='*60}")
    print(f"Images processed: {total_images}")
    print(f"Total matras detected: {total_detections}")
    print(f"Average per image: {total_detections/total_images:.1f}")
    
    # Count by class
    class_totals = {}
    for pred_set in all_predictions:
        for pred in pred_set['predictions']:
            cls = pred['class']
            class_totals[cls] = class_totals.get(cls, 0) + 1
    
    if class_totals:
        print(f"\nDetections by matra type:")
        for cls, count in sorted(class_totals.items()):
            print(f"  {cls}: {count}")
    
    print(f"\nVisualizations saved to: {save_dir.absolute()}")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(
        description='Predict Modi matras using trained YOLO model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python predict_modi_matra.py --model best.pt --source test.jpg
  
  # Folder of images
  python predict_modi_matra.py --model best.pt --source test_images/ --conf 0.3
  
  # Save predictions as JSON
  python predict_modi_matra.py --model best.pt --source test/ --json
  
  # Custom output directory
  python predict_modi_matra.py --model best.pt --source test/ --output my_predictions/
        """
    )
    
    parser.add_argument('--model', required=True,
                       help='Path to trained model (e.g., runs/train/weights/best.pt)')
    parser.add_argument('--source', required=True,
                       help='Image file or directory to process')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--output', default='predictions',
                       help='Output directory (default: predictions)')
    parser.add_argument('--json', action='store_true',
                       help='Save predictions as JSON file')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.model).exists():
        print(f"ERROR: Model not found: {args.model}")
        return
    
    if not Path(args.source).exists():
        print(f"ERROR: Source not found: {args.source}")
        return
    
    # Run prediction
    predict_and_visualize(
        model_path=args.model,
        source=args.source,
        conf=args.conf,
        save_dir=args.output,
        save_json=args.json
    )

if __name__ == '__main__':
    main()
