#!/usr/bin/env python3
"""
COMPARE ORIGINAL vs STAGE 4 MODEL
==================================

Compares layer-by-layer analysis between:
- Original baseline model (96.4% mAP)
- Stage 4 optimized model (96.3% mAP)

Shows which layers improved, which got worse, and overall impact.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_analysis(json_path):
    """Load layer analysis JSON"""
    with open(json_path, 'r') as f:
        return json.load(f)

def classify_predictions(data):
    """Classify images as GOOD or BAD predictions"""
    good_images = []
    bad_images = []
    
    for result in data['detailed_results']:
        predictions = result.get('predictions', [])
        if not predictions:
            bad_images.append(result)
            continue
        
        max_confidence = max([p['confidence'] for p in predictions])
        
        if max_confidence >= 0.85:
            good_images.append(result)
        else:
            bad_images.append(result)
    
    return good_images, bad_images

def extract_layer_stats(images, layer_name):
    """Extract statistics for a specific layer across images"""
    stats = {
        'means': [],
        'stds': [],
        'sparsities': []
    }
    
    for img in images:
        layer_outputs = img.get('layer_outputs', {})
        for key, value in layer_outputs.items():
            if layer_name in key and isinstance(value, dict):
                stats['means'].append(value.get('mean', 0))
                stats['stds'].append(value.get('std', 0))
                stats['sparsities'].append(value.get('sparsity', 0))
    
    return {
        'mean': np.mean(stats['means']) if stats['means'] else 0,
        'std': np.mean(stats['stds']) if stats['stds'] else 0,
        'sparsity': np.mean(stats['sparsities']) if stats['sparsities'] else 0
    }

def compare_models(original_json, stage4_json, output_dir='comparison_results'):
    """
    Compare two models' layer analyses
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("COMPARING ORIGINAL vs STAGE 4 MODEL")
    print("="*80)
    
    # Load data
    print("\n📂 Loading analysis data...")
    original_data = load_analysis(original_json)
    stage4_data = load_analysis(stage4_json)
    
    # Classify predictions for both models
    print("📊 Classifying predictions...")
    orig_good, orig_bad = classify_predictions(original_data)
    s4_good, s4_bad = classify_predictions(stage4_data)
    
    print(f"\nORIGINAL MODEL:")
    print(f"  GOOD predictions: {len(orig_good)} ({len(orig_good)/original_data['total_images']*100:.1f}%)")
    print(f"  BAD predictions:  {len(orig_bad)} ({len(orig_bad)/original_data['total_images']*100:.1f}%)")
    
    print(f"\nSTAGE 4 MODEL:")
    print(f"  GOOD predictions: {len(s4_good)} ({len(s4_good)/stage4_data['total_images']*100:.1f}%)")
    print(f"  BAD predictions:  {len(s4_bad)} ({len(s4_bad)/stage4_data['total_images']*100:.1f}%)")
    
    # Extract layer names
    sample_img = original_data['detailed_results'][0]
    layer_names = [k for k in sample_img['layer_outputs'].keys() 
                   if 'Conv' in k or 'C2f' in k][:10]  # Analyze key layers
    
    # Compare layers
    print("\n📊 Comparing layer activations...")
    comparison_data = []
    
    for layer in layer_names:
        # Original model
        orig_good_stats = extract_layer_stats(orig_good, layer)
        orig_bad_stats = extract_layer_stats(orig_bad, layer)
        orig_diff = abs(orig_bad_stats['mean'] - orig_good_stats['mean'])
        
        # Stage 4 model  
        s4_good_stats = extract_layer_stats(s4_good, layer)
        s4_bad_stats = extract_layer_stats(s4_bad, layer)
        s4_diff = abs(s4_bad_stats['mean'] - s4_good_stats['mean'])
        
        comparison_data.append({
            'layer': layer,
            'original_diff': orig_diff,
            'stage4_diff': s4_diff,
            'improvement': ((orig_diff - s4_diff) / orig_diff * 100) if orig_diff > 0 else 0,
            'orig_sparsity': orig_bad_stats['sparsity'],
            's4_sparsity': s4_bad_stats['sparsity']
        })
    
    # Sort by improvement
    comparison_data.sort(key=lambda x: abs(x['improvement']), reverse=True)
    
    # Print summary
    print("\n" + "="*80)
    print("LAYER IMPROVEMENT SUMMARY")
    print("="*80)
    print(f"{'Layer':<25} {'Original Diff':<15} {'Stage4 Diff':<15} {'Improvement':<15}")
    print("-"*80)
    
    for item in comparison_data:
        improvement_str = f"{item['improvement']:+.1f}%"
        status = "✅" if item['improvement'] > 5 else "➖" if abs(item['improvement']) < 5 else "❌"
        print(f"{item['layer']:<25} {item['original_diff']:<15.4f} {item['stage4_diff']:<15.4f} {improvement_str:<12} {status}")
    
    # Overall comparison
    print("\n" + "="*80)
    print("OVERALL COMPARISON")
    print("="*80)
    
    orig_success_rate = len(orig_good) / original_data['total_images'] * 100
    s4_success_rate = len(s4_good) / stage4_data['total_images'] * 100
    success_change = s4_success_rate - orig_success_rate
    
    print(f"\nSuccess Rate (confidence >= 0.85):")
    print(f"  Original: {orig_success_rate:.1f}%")
    print(f"  Stage 4:  {s4_success_rate:.1f}%")
    print(f"  Change:   {success_change:+.1f}% {'✅ BETTER' if success_change > 0 else '❌ WORSE' if success_change < 0 else '➖ SAME'}")
    
    print(f"\nFailed Predictions:")
    print(f"  Original: {len(orig_bad)} images")
    print(f"  Stage 4:  {len(s4_bad)} images")
    print(f"  Change:   {len(s4_bad) - len(orig_bad):+d} images")
    
    # Calculate layer stability improvement
    improved_layers = [x for x in comparison_data if x['improvement'] > 5]
    degraded_layers = [x for x in comparison_data if x['improvement'] < -5]
    
    print(f"\nLayer Stability:")
    print(f"  Improved layers:  {len(improved_layers)}")
    print(f"  Degraded layers:  {len(degraded_layers)}")
    print(f"  Unchanged layers: {len(comparison_data) - len(improved_layers) - len(degraded_layers)}")
    
    # Create visualization
    print("\n📊 Creating comparison charts...")
    create_comparison_charts(comparison_data, orig_success_rate, s4_success_rate, 
                            len(orig_bad), len(s4_bad), output_dir)
    
    # Save detailed report
    with open(output_dir / 'comparison_report.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("ORIGINAL vs STAGE 4 MODEL COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("PREDICTION QUALITY:\n")
        f.write("-"*80 + "\n")
        f.write(f"Original Model:  {len(orig_good)} good, {len(orig_bad)} bad ({orig_success_rate:.1f}% success)\n")
        f.write(f"Stage 4 Model:   {len(s4_good)} good, {len(s4_bad)} bad ({s4_success_rate:.1f}% success)\n")
        f.write(f"Change:          {success_change:+.1f}% success rate\n\n")
        
        f.write("LAYER-BY-LAYER COMPARISON:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Layer':<25} {'Orig Diff':<12} {'S4 Diff':<12} {'Improvement':<12} Status\n")
        f.write("-"*80 + "\n")
        
        for item in comparison_data:
            status = "IMPROVED" if item['improvement'] > 5 else "UNCHANGED" if abs(item['improvement']) < 5 else "DEGRADED"
            f.write(f"{item['layer']:<25} {item['original_diff']:<12.4f} {item['stage4_diff']:<12.4f} {item['improvement']:+11.1f}% {status}\n")
    
    print(f"\n✅ Comparison complete!")
    print(f"📁 Results saved to: {output_dir}/")
    print(f"   - comparison_report.txt")
    print(f"   - comparison_charts.png")
    print("="*80)
    
    return comparison_data

def create_comparison_charts(comparison_data, orig_success, s4_success, 
                            orig_bad_count, s4_bad_count, output_dir):
    """Create visualization comparing the two models"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Original vs Stage 4 Model Comparison', fontsize=16, fontweight='bold')
    
    # Chart 1: Layer activation differences
    ax1 = axes[0, 0]
    layers = [x['layer'].replace('Layer_', 'L').replace('_Conv', '') for x in comparison_data[:8]]
    orig_diffs = [x['original_diff'] for x in comparison_data[:8]]
    s4_diffs = [x['stage4_diff'] for x in comparison_data[:8]]
    
    x = np.arange(len(layers))
    width = 0.35
    
    ax1.bar(x - width/2, orig_diffs, width, label='Original', color='#ff6b6b', alpha=0.8)
    ax1.bar(x + width/2, s4_diffs, width, label='Stage 4', color='#4ecdc4', alpha=0.8)
    ax1.set_xlabel('Layer', fontweight='bold')
    ax1.set_ylabel('Activation Difference (Good vs Bad)', fontweight='bold')
    ax1.set_title('Layer Activation Differences', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Chart 2: Improvement percentage
    ax2 = axes[0, 1]
    improvements = [x['improvement'] for x in comparison_data[:8]]
    colors = ['#51cf66' if x > 0 else '#ff6b6b' for x in improvements]
    
    ax2.barh(layers, improvements, color=colors, alpha=0.8)
    ax2.set_xlabel('Improvement (%)', fontweight='bold')
    ax2.set_title('Layer Improvement After Stage 4 Optimization', fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add percentage labels
    for i, (layer, imp) in enumerate(zip(layers, improvements)):
        ax2.text(imp + 1 if imp > 0 else imp - 1, i, f'{imp:+.1f}%', 
                va='center', ha='left' if imp > 0 else 'right', fontsize=9)
    
    # Chart 3: Success rate comparison
    ax3 = axes[1, 0]
    models = ['Original\nModel', 'Stage 4\nModel']
    success_rates = [orig_success, s4_success]
    colors_success = ['#ff6b6b' if s4_success < orig_success else '#51cf66' 
                     for _ in success_rates]
    
    bars = ax3.bar(models, success_rates, color=colors_success, alpha=0.8, width=0.6)
    ax3.set_ylabel('Success Rate (%)', fontweight='bold')
    ax3.set_title('Overall Success Rate Comparison', fontweight='bold')
    ax3.set_ylim([85, 100])
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, success_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Chart 4: Failed predictions count
    ax4 = axes[1, 1]
    bad_counts = [orig_bad_count, s4_bad_count]
    colors_bad = ['#ff6b6b' if s4_bad_count > orig_bad_count else '#51cf66' 
                 for _ in bad_counts]
    
    bars = ax4.bar(models, bad_counts, color=colors_bad, alpha=0.8, width=0.6)
    ax4.set_ylabel('Number of Failed Predictions', fontweight='bold')
    ax4.set_title('Failed Predictions Count', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, bad_counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(val)}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add change indicator
    change = s4_bad_count - orig_bad_count
    change_text = f"Change: {change:+d} ({change/orig_bad_count*100:+.1f}%)"
    ax4.text(0.5, 0.95, change_text, transform=ax4.transAxes,
            ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_charts.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare Original vs Stage 4 Model')
    parser.add_argument('--original', default='layer_analysis_ORIGINAL/layer_analysis_report.json',
                       help='Path to original model analysis JSON')
    parser.add_argument('--stage4', default='layer_analysis_STAGE4/layer_analysis_report.json',
                       help='Path to stage 4 model analysis JSON')
    parser.add_argument('--output', default='comparison_results',
                       help='Output directory for comparison results')
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.original).exists():
        print(f"❌ ERROR: Original analysis not found: {args.original}")
        print(f"\nRun this first:")
        print(f"  python analyze_yolo_layers.py \\")
        print(f"      --model runs/modi_matra/train_full_7k2/weights/best.pt \\")
        print(f"      --images modi_full_7k/merged/images/test/ \\")
        print(f"      --output layer_analysis_ORIGINAL")
        return
    
    if not Path(args.stage4).exists():
        print(f"❌ ERROR: Stage 4 analysis not found: {args.stage4}")
        print(f"\nRun this first:")
        print(f"  python analyze_yolo_layers.py \\")
        print(f"      --model runs/progressive_fix/stage4_final/weights/best.pt \\")
        print(f"      --images modi_full_7k/merged/images/test/ \\")
        print(f"      --output layer_analysis_STAGE4")
        return
    
    # Run comparison
    compare_models(args.original, args.stage4, args.output)

if __name__ == '__main__':
    main()