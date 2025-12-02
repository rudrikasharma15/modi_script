#!/usr/bin/env python3
"""
analyze_yolo_layers.py
Extracts layer-by-layer activations from YOLO model during inference
"""
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
from tqdm import tqdm
import argparse

class YOLOLayerAnalyzer:
    """
    Analyzes YOLO model layer by layer
    """
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.model.model.eval()
        self.layer_outputs = {}
        self.hooks = []
        
        print(f"Loading model: {model_path}")
        print("Model loaded successfully")
        
    def register_hooks(self):
        """Register forward hooks on all layers"""
        
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, torch.Tensor):
                    self.layer_outputs[name] = output.detach().cpu()
                elif isinstance(output, (list, tuple)):
                    # For detection heads that return multiple outputs
                    self.layer_outputs[name] = [o.detach().cpu() if isinstance(o, torch.Tensor) else o for o in output]
            return hook
        
        # Register hooks on all model layers
        layer_count = 0
        for idx, layer in enumerate(self.model.model.model):
            layer_name = f"Layer_{idx}_{layer.__class__.__name__}"
            handle = layer.register_forward_hook(get_activation(layer_name))
            self.hooks.append(handle)
            layer_count += 1
        
        print(f"Registering hooks on {layer_count} layers...")
        
    def analyze_tensor(self, tensor):
        """Compute statistics for a tensor"""
        if not isinstance(tensor, torch.Tensor):
            return None
        
        arr = tensor.numpy()
        
        return {
            'shape': list(arr.shape),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'sparsity': float(np.mean(arr == 0))  # Percentage of dead neurons
        }
    
    def process_image(self, image_path):
        """Process single image and collect layer outputs"""
        self.layer_outputs = {}
        
        # Run inference
        results = self.model.predict(image_path, verbose=False)
        
        # Analyze each layer's output
        layer_stats = {}
        for layer_name, output in self.layer_outputs.items():
            if isinstance(output, torch.Tensor):
                stats = self.analyze_tensor(output)
                if stats:
                    layer_stats[layer_name] = stats
            elif isinstance(output, (list, tuple)):
                # Handle detection heads
                layer_stats[layer_name] = []
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor):
                        stats = self.analyze_tensor(o)
                        if stats:
                            layer_stats[layer_name].append(stats)
        
        # Get predictions
        predictions = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                predictions.append({
                    'class': int(boxes.cls[i].item()),
                    'confidence': float(boxes.conf[i].item()),
                    'bbox': boxes.xyxy[i].cpu().numpy().tolist()
                })
        
        return {
            'image': str(image_path),
            'predictions': predictions,
            'layer_outputs': layer_stats
        }
    
    def analyze_dataset(self, image_dir, max_images=None):
        """Analyze all images in directory"""
        
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"Processing {len(image_files)} test images...")
        
        results = []
        for img_path in tqdm(image_files, desc="Analyzing images"):
            try:
                result = self.process_image(img_path)
                results.append(result)
            except Exception as e:
                print(f"\n⚠️  Error processing {img_path}: {e}")
                continue
        
        return results
    
    def save_results(self, results, output_dir='layer_analysis_output'):
        """Save analysis results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        report = {
            'total_images': len(results),
            'detailed_results': results
        }
        
        output_file = output_dir / 'layer_analysis_report.json'
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}")
        
        return output_file
    
    def cleanup(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()

def main():
    parser = argparse.ArgumentParser(
        description='Analyze YOLO layer activations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all test images
  python analyze_yolo_layers.py --model best.pt --images test/ --output layer_analysis_after
  
  # Analyze first 50 images only
  python analyze_yolo_layers.py --model best.pt --images test/ --output layer_analysis_after --max 50
        """
    )
    
    parser.add_argument('--model', required=True,
                       help='Path to YOLO model (.pt file)')
    parser.add_argument('--images', required=True,
                       help='Path to test images directory')
    parser.add_argument('--output', default='layer_analysis_output',
                       help='Output directory (default: layer_analysis_output)')
    parser.add_argument('--max', type=int, default=None,
                       help='Maximum number of images to analyze (default: all)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.model).exists():
        print(f"❌ ERROR: Model not found: {args.model}")
        return
    
    if not Path(args.images).exists():
        print(f"❌ ERROR: Image directory not found: {args.images}")
        return
    
    print("\n" + "="*80)
    print("YOLO LAYER-BY-LAYER ANALYSIS")
    print("="*80 + "\n")
    
    # Create analyzer
    analyzer = YOLOLayerAnalyzer(args.model)
    
    # Register hooks
    analyzer.register_hooks()
    
    # Analyze dataset
    results = analyzer.analyze_dataset(args.images, max_images=args.max)
    
    # Save results
    output_file = analyzer.save_results(results, args.output)
    
    # Cleanup
    analyzer.cleanup()
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nNext step: Run comparison analysis")
    print(f"  python analyze_layers.py")
    print(f"\nOr compare before/after:")
    print(f"  python compare_before_after.py \\")
    print(f"    --before layer_analysis_report_BEFORE.json \\")
    print(f"    --after {args.output}/layer_analysis_report.json")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()