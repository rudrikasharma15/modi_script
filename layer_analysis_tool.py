#!/usr/bin/env python3
"""
YOLOv8 Layer-by-Layer Output Analysis
Shows what each layer produces and identifies problematic layers

FIXED: Now analyzes ALL test images (not just 20)
"""

import torch
import numpy as np
from ultralytics import YOLO
import cv2
from pathlib import Path
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

class LayerAnalyzer:
    """
    Analyzes each layer's output in YOLOv8 to identify which layers
    are causing wrong predictions.
    """
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.layer_outputs = {}
        self.hooks = []

    def register_hooks(self):
        """
        Register forward hooks to capture each layer's output.
        """
        print("\n" + "="*80)
        print("REGISTERING HOOKS ON ALL LAYERS")
        print("="*80)

        def get_activation(name):
            def hook(model, input, output):
                # If output is a tuple, detach each tensor
                if isinstance(output, tuple):
                    self.layer_outputs[name] = tuple(
                        o.detach().cpu() if isinstance(o, torch.Tensor) else o for o in output
                    )
                elif isinstance(output, torch.Tensor):
                    self.layer_outputs[name] = output.detach().cpu()
                else:
                    self.layer_outputs[name] = output  # fallback for other types
            return hook

        # Register hook on every layer
        for idx, layer in enumerate(self.model.model.model):
            layer_name = f"Layer_{idx}_{type(layer).__name__}"
            hook = layer.register_forward_hook(get_activation(layer_name))
            self.hooks.append(hook)
            print(f"  ✅ Registered: {layer_name}")

        print(f"\n📊 Total layers monitored: {len(self.hooks)}")
        print("="*80)

    def analyze_image(self, image_path, ground_truth_boxes=None):
        """
        Run image through model and capture all layer outputs.
        """
        # Clear previous outputs
        self.layer_outputs = {}
        
        # Run inference
        results = self.model.predict(image_path, conf=0.25, verbose=False)[0]
        
        # Get predictions
        predictions = []
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            predictions.append({
                'class': cls,
                'confidence': float(conf),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })
        
        # Analyze each layer
        layer_analysis = {}
        for layer_name, output in self.layer_outputs.items():
            analysis = self._analyze_layer_output(layer_name, output)
            layer_analysis[layer_name] = analysis
        
        # Compare with ground truth
        comparison = None
        if ground_truth_boxes:
            comparison = self._compare_with_ground_truth(predictions, ground_truth_boxes)
        
        return {
            'image': str(image_path),
            'predictions': predictions,
            'ground_truth': ground_truth_boxes,
            'comparison': comparison,
            'layer_outputs': layer_analysis
        }

    def _analyze_layer_output(self, layer_name, output):
        """
        Analyze a single layer's output.
        Converts all numeric values to Python float for JSON serialization.
        """
        if isinstance(output, torch.Tensor):
            out_np = output.cpu().numpy()
            return {
                'shape': list(out_np.shape),
                'mean': float(out_np.mean()),
                'std': float(out_np.std()),
                'min': float(out_np.min()),
                'max': float(out_np.max()),
                'has_nan': bool(np.isnan(out_np).any()),
                'has_inf': bool(np.isinf(out_np).any()),
                'sparsity': float((out_np == 0).mean()),  # % of zeros
            }
        elif isinstance(output, tuple):
            return tuple(self._analyze_layer_output(f"{layer_name}_tuple_{i}", o) if isinstance(o, torch.Tensor) else str(type(o)) for i, o in enumerate(output))
        else:
            return {'type': str(type(output))}

    def _compare_with_ground_truth(self, predictions, ground_truth):
        """
        Compare predictions with ground truth to identify errors.
        """
        tp, fp, fn = 0, 0, 0
        matched_gt = set()
        
        for pred in predictions:
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(ground_truth):
                iou = self._calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou > 0.5 and best_gt_idx >= 0:
                if pred['class'] == ground_truth[best_gt_idx]['class']:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1
            else:
                fp += 1
        
        fn = len(ground_truth) - len(matched_gt)
        
        return {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0
        }

    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes [x1, y1, x2, y2]."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x1 = max(x1_min, x2_min)
        inter_y1 = max(y1_min, y2_min)
        inter_x2 = min(x1_max, x2_max)
        inter_y2 = min(y1_max, y2_max)
        
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return float(inter_area / union_area) if union_area > 0 else 0.0

    def identify_problematic_layers(self, test_images, output_dir='layer_analysis', max_images=None):
        """
        Test on multiple images and identify which layers cause wrong outputs.
        
        Args:
            test_images: List of image paths
            output_dir: Where to save results
            max_images: Maximum number of images to analyze (None = all)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print("IDENTIFYING PROBLEMATIC LAYERS")
        print("="*80)
        
        # Limit images if specified
        if max_images:
            test_images = test_images[:max_images]
            print(f"📊 Analyzing {len(test_images)} images (limited to {max_images})")
        else:
            print(f"📊 Analyzing ALL {len(test_images)} images")
        
        all_results = []
        layer_error_counts = defaultdict(int)
        
        # Use tqdm for progress bar
        for img_path in tqdm(test_images, desc="Analyzing images"):
            result = self.analyze_image(img_path)
            all_results.append(result)
            
            if result['comparison']:
                if result['comparison']['precision'] < 1.0 or result['comparison']['recall'] < 1.0:
                    for layer_name, stats in result['layer_outputs'].items():
                        if isinstance(stats, dict):
                            if stats.get('has_nan', False):
                                layer_error_counts[f"{layer_name}_NaN"] += 1
                            if stats.get('has_inf', False):
                                layer_error_counts[f"{layer_name}_Inf"] += 1
                            if stats.get('sparsity', 0) > 0.9:
                                layer_error_counts[f"{layer_name}_Dead"] += 1
        
        report = {
            'total_images': len(test_images),
            'problematic_layers': dict(layer_error_counts),
            'detailed_results': all_results
        }
        
        with open(output_dir / 'layer_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\n📊 PROBLEMATIC LAYERS SUMMARY:")
        print("="*80)
        if layer_error_counts:
            sorted_layers = sorted(layer_error_counts.items(), key=lambda x: x[1], reverse=True)
            for layer_name, count in sorted_layers[:10]:
                print(f"  {layer_name}: {count} errors")
        else:
            print("  ✅ No problematic layers detected!")
        print(f"\n📁 Full report saved to: {output_dir / 'layer_analysis_report.json'}")
        print("="*80)
        
        return report

    def visualize_layer_activations(self, image_path, layer_indices=[0, 5, 10, 15, 20], output_dir='layer_viz'):
        """
        Visualize what specific layers are seeing/producing.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n🎨 Visualizing layer activations for: {Path(image_path).name}")
        
        self.analyze_image(image_path)
        
        for idx in layer_indices:
            layer_name = f"Layer_{idx}"
            matching_layers = [k for k in self.layer_outputs.keys() if k.startswith(layer_name)]
            if not matching_layers:
                continue
            
            layer_name = matching_layers[0]
            output = self.layer_outputs[layer_name]
            if not isinstance(output, torch.Tensor):
                continue
            
            activations = output[0][:16].cpu().numpy()
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            fig.suptitle(f'{layer_name} - Feature Maps', fontsize=16)
            
            for i, ax in enumerate(axes.flat):
                if i < len(activations):
                    act = activations[i]
                    act = (act - act.min()) / (act.max() - act.min() + 1e-8)
                    ax.imshow(act, cmap='viridis')
                    ax.axis('off')
                    ax.set_title(f'Ch {i}', fontsize=8)
                else:
                    ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{layer_name}_activation.png', dpi=150)
            plt.close()
            print(f"  ✅ Saved: {layer_name}_activation.png")
        
        print(f"\n📁 Visualizations saved to: {output_dir}")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        print("\n✅ All hooks removed")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='YOLOv8 Layer-by-Layer Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze ALL test images
  python analyze_yolo_layers.py --model best.pt --images test_images/
  
  # Limit to first 50 images
  python analyze_yolo_layers.py --model best.pt --images test_images/ --max 50
  
  # With visualizations
  python analyze_yolo_layers.py --model best.pt --images test/ --visualize
        """
    )
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--images', required=True, help='Directory with test images')
    parser.add_argument('--output', default='layer_analysis', help='Output directory')
    parser.add_argument('--max', type=int, default=None, 
                       help='Maximum number of images to analyze (default: all)')
    parser.add_argument('--visualize', action='store_true', 
                       help='Visualize layer activations for first image')
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.model).exists():
        print(f"❌ ERROR: Model not found: {args.model}")
        return
    
    image_dir = Path(args.images)
    if not image_dir.exists():
        print(f"❌ ERROR: Image directory not found: {args.images}")
        return
    
    # Find all images
    test_images = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')) + \
                  list(image_dir.glob('*.jpeg')) + list(image_dir.glob('*.JPG')) + \
                  list(image_dir.glob('*.PNG'))
    
    if not test_images:
        print(f"❌ ERROR: No images found in: {args.images}")
        return
    
    print(f"\n{'='*80}")
    print(f"📸 Found {len(test_images)} test images")
    if args.max:
        print(f"⚠️  Will analyze first {args.max} images only")
    else:
        print(f"✅ Will analyze ALL {len(test_images)} images")
    print(f"{'='*80}")
    
    # Create analyzer and register hooks
    analyzer = LayerAnalyzer(args.model)
    analyzer.register_hooks()
    
    # Run analysis
    report = analyzer.identify_problematic_layers(
        test_images, 
        args.output,
        max_images=args.max
    )
    
    # Visualize if requested
    if args.visualize and test_images:
        analyzer.visualize_layer_activations(
            str(test_images[0]),
            layer_indices=[0, 3, 6, 9, 12, 15, 18, 21],
            output_dir=Path(args.output) / 'visualizations'
        )
    
    # Cleanup
    analyzer.remove_hooks()
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n📋 Summary:")
    print(f"  Images analyzed: {report['total_images']}")
    print(f"  Total layers: {len(analyzer.layer_outputs)}")
    print(f"  Problematic layers: {len(report['problematic_layers'])}")
    print(f"\n📁 Results saved to: {args.output}")
    print(f"\n🔍 Next step: Run analysis on this JSON:")
    print(f"  python analyze_layers.py")
    print("="*80)


if __name__ == '__main__':
    main()