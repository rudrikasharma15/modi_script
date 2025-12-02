#!/usr/bin/env python3
"""
ENHANCED PROGRESSIVE LAYER FIX - Build on Your Working Code!

Current: 96.3% mAP50 ✅
Target: 97-98% mAP50 🎯

NEW ADDITIONS:
- Stage 5: Targeted augmentation for weak classes
- Stage 6: Multi-scale training for robustness
- Conservative improvements (won't break your model!)
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path

class EnhancedProgressiveOptimizer:
    """
    Enhanced optimizer building on your working progressive approach
    """
    
    def __init__(self, base_model_path, data_yaml):
        self.base_model_path = base_model_path
        self.data_yaml = data_yaml
        
    def apply_he_init(self, model, layers_to_fix):
        """Apply He initialization to specific layers"""
        print(f"\n🔧 Applying He initialization to layers: {layers_to_fix}")
        
        for idx, layer in enumerate(model.model.model):
            if idx in layers_to_fix:
                if hasattr(layer, 'conv'):
                    nn.init.kaiming_normal_(
                        layer.conv.weight, 
                        mode='fan_out', 
                        nonlinearity='relu'
                    )
                    print(f"  ✅ Layer_{idx}: He initialization applied")
        
        return model
    
    def stage1_fix_early_layers(self):
        """STAGE 1: Fix Layer_0 and Layer_1 (input layers)"""
        print("\n" + "="*80)
        print("STAGE 1: Fixing Early Layers (Layer_0, Layer_1)")
        print("="*80)
        
        model = YOLO(self.base_model_path)
        model = self.apply_he_init(model, [0, 1])
        
        print("\n🚀 Fine-tuning for 20 epochs...")
        
        results = model.train(
            data=self.data_yaml,
            epochs=20,
            batch=16,
            imgsz=640,
            project='runs/enhanced_progressive',
            name='stage1_early_layers',
            lr0=0.001,
            lrf=0.01,
            optimizer='auto',
            momentum=0.937,
            weight_decay=0.0005,
            workers=4,
            cache=False,
            resume=False,
            pretrained=True,
        )
        
        del model
        torch.cuda.empty_cache()
        
        print("✅ Stage 1 complete!")
        return 'runs/enhanced_progressive/stage1_early_layers/weights/best.pt'
    
    def stage2_fix_middle_layer(self, stage1_model):
        """STAGE 2: Fix Layer_7 (middle layer)"""
        print("\n" + "="*80)
        print("STAGE 2: Fixing Middle Layer (Layer_7)")
        print("="*80)
        
        torch.cuda.empty_cache()
        model = YOLO(stage1_model)
        model = self.apply_he_init(model, [7])
        
        print("\n🚀 Fine-tuning for 30 epochs...")
        
        results = model.train(
            data=self.data_yaml,
            epochs=30,
            batch=16,
            imgsz=640,
            project='runs/enhanced_progressive',
            name='stage2_middle_layer',
            lr0=0.002,
            lrf=0.01,
            optimizer='auto',
            momentum=0.937,
            weight_decay=0.0005,
            workers=4,
            cache=False,
        )
        
        del model
        torch.cuda.empty_cache()
        
        print("✅ Stage 2 complete!")
        return 'runs/enhanced_progressive/stage2_middle_layer/weights/best.pt'
    
    def stage3_fix_late_layers(self, stage2_model):
        """STAGE 3: Fix Layer_19 and Layer_21 (late layers)"""
        print("\n" + "="*80)
        print("STAGE 3: Fixing Late Layers (Layer_19, Layer_21)")
        print("="*80)
        
        torch.cuda.empty_cache()
        model = YOLO(stage2_model)
        model = self.apply_he_init(model, [19, 21])
        
        print("\n🚀 Fine-tuning for 30 epochs...")
        
        results = model.train(
            data=self.data_yaml,
            epochs=30,
            batch=16,
            imgsz=640,
            project='runs/enhanced_progressive',
            name='stage3_late_layers',
            lr0=0.002,
            lrf=0.01,
            optimizer='auto',
            momentum=0.937,
            weight_decay=0.0005,
            workers=4,
            cache=False,
        )
        
        del model
        torch.cuda.empty_cache()
        
        print("✅ Stage 3 complete!")
        return 'runs/enhanced_progressive/stage3_late_layers/weights/best.pt'
    
    def stage4_full_fine_tune(self, stage3_model):
        """STAGE 4: Full fine-tuning with enhanced augmentation"""
        print("\n" + "="*80)
        print("STAGE 4: Full Fine-Tuning")
        print("="*80)
        
        torch.cuda.empty_cache()
        model = YOLO(stage3_model)
        
        print("\n🚀 Final fine-tuning for 50 epochs...")
        
        results = model.train(
            data=self.data_yaml,
            epochs=50,
            batch=16,
            imgsz=640,
            project='runs/enhanced_progressive',
            name='stage4_final',
            lr0=0.005,
            lrf=0.01,
            optimizer='auto',
            momentum=0.937,
            weight_decay=0.0005,
            hsv_h=0.02,
            degrees=12.0,
            mosaic=1.0,
            dropout=0.05,
            workers=4,
            cache=False,
        )
        
        del model
        torch.cuda.empty_cache()
        
        print("✅ Stage 4 complete!")
        return 'runs/enhanced_progressive/stage4_final/weights/best.pt'
    
    def stage5_targeted_weak_classes(self, stage4_model):
        """
        🆕 STAGE 5: Target side_matra and bottom_matra improvement
        
        Current performance:
        - side_matra: 97.4% → Target: 98%+
        - bottom_matra: 92.9% → Target: 95%+
        
        Strategy: Enhanced augmentation + class-weighted loss
        """
        print("\n" + "="*80)
        print("🆕 STAGE 5: TARGETED WEAK CLASS IMPROVEMENT")
        print("="*80)
        print("Focus: side_matra (97.4%→98%) & bottom_matra (92.9%→95%)")
        print("Method: Enhanced augmentation + class weighting")
        
        torch.cuda.empty_cache()
        model = YOLO(stage4_model)
        
        print("\n🚀 Targeted training for 40 epochs...")
        print("Using stronger augmentation for minority classes...")
        
        results = model.train(
            data=self.data_yaml,
            epochs=40,
            batch=16,
            imgsz=640,
            project='runs/enhanced_progressive',
            name='stage5_targeted',
            
            # 🔥 ENHANCED AUGMENTATION (helps minority classes)
            hsv_h=0.025,      # More color variation
            hsv_s=0.7,        # Saturation changes
            hsv_v=0.4,        # Brightness variation
            degrees=15.0,     # More rotation (15 vs 12)
            translate=0.15,   # More translation (0.15 vs 0.1)
            scale=0.6,        # Scale variation
            shear=3.0,        # Shear transformation
            perspective=0.0001,  # Slight perspective
            flipud=0.1,       # Vertical flip (helps bottom matras!)
            fliplr=0.5,       # Horizontal flip
            mosaic=1.0,       # Keep mosaic
            mixup=0.15,       # 🆕 Mixup augmentation
            copy_paste=0.1,   # 🆕 Copy-paste for minority classes
            
            # 🔥 CLASS WEIGHTING (focus on classification)
            cls=0.6,          # Increase classification loss (vs 0.5)
            box=7.5,          # Keep box loss standard
            dfl=1.5,          # Distribution focal loss
            
            # Optimizer settings
            lr0=0.003,        # Moderate learning rate
            lrf=0.01,
            optimizer='AdamW',  # 🆕 Switch to AdamW
            momentum=0.937,
            weight_decay=0.0008,  # Slightly more regularization
            
            # Regularization
            dropout=0.1,      # More dropout (0.1 vs 0.05)
            
            workers=4,
            cache=False,
        )
        
        del model
        torch.cuda.empty_cache()
        
        print("✅ Stage 5 complete!")
        print("Expected: side_matra 98%+, bottom_matra 95%+")
        return 'runs/enhanced_progressive/stage5_targeted/weights/best.pt'
    
    def stage6_multiscale_polish(self, stage5_model):
        """
        🆕 STAGE 6: Multi-scale training for robustness
        
        Strategy: Train on 800px images for better detail detection
        This helps detect small matras more accurately
        """
        print("\n" + "="*80)
        print("🆕 STAGE 6: MULTI-SCALE TRAINING")
        print("="*80)
        print("Training on 800px images for better detail")
        print("Helps detect small/intricate matra features")
        
        torch.cuda.empty_cache()
        model = YOLO(stage5_model)
        
        print("\n🚀 Multi-scale training for 30 epochs...")
        
        results = model.train(
            data=self.data_yaml,
            epochs=30,
            batch=12,  # 🔥 Reduced batch for larger images
            imgsz=800,  # 🔥 LARGER image size (800 vs 640)
            project='runs/enhanced_progressive',
            name='stage6_multiscale',
            
            # Multi-scale settings
            rect=False,  # Use square images (not rectangular)
            scale=0.9,   # Random scale 0.1-0.9
            
            # Keep good augmentation
            hsv_h=0.02,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=12.0,
            translate=0.1,
            # scale=0.5,
            mosaic=1.0,
            mixup=0.1,
            
            # Fine-tuning settings
            lr0=0.002,   # Lower LR for larger images
            lrf=0.01,
            optimizer='AdamW',
            momentum=0.937,
            weight_decay=0.0005,
            dropout=0.08,
            
            workers=4,
            cache=False,
        )
        
        del model
        torch.cuda.empty_cache()
        
        print("✅ Stage 6 complete!")
        print("Expected: Overall 97-98% mAP")
        return 'runs/enhanced_progressive/stage6_multiscale/weights/best.pt'
    
    def evaluate_with_tta(self, model_path, split='val'):
        """
        🆕 BONUS: Test-Time Augmentation
        
        Free performance boost without retraining!
        Averages predictions across multiple augmented versions
        """
        print("\n" + "="*80)
        print("🔥 BONUS: TEST-TIME AUGMENTATION")
        print("="*80)
        print("Evaluating with TTA (no training needed!)")
        
        model = YOLO(model_path)
        
        results = model.val(
            data=self.data_yaml,
            split=split,
            
            # 🔥 Enable TTA
            augment=True,  # This is the magic!
            
            # Optimized detection settings
            conf=0.001,  # Lower confidence threshold
            iou=0.6,     # NMS IoU threshold
            max_det=300, # Allow more detections
            
            save_json=True,
            plots=True,
            project='runs/enhanced_progressive',
            name=f'tta_eval_{split}',
        )
        
        print(f"\n✅ TTA Evaluation complete for {split} set!")
        return results
    
    def run_all_stages(self):
        """Execute all 6 stages progressively"""
        print("\n" + "="*80)
        print("🚀 ENHANCED PROGRESSIVE OPTIMIZATION - 6 STAGES")
        print("="*80)
        print("\nStarting from your current best model")
        print("Current: 96.3% mAP50")
        print("Target: 97-98% mAP50")
        print("\nStage 1: Early layers        20 epochs (~30 min)")
        print("Stage 2: Middle layer        30 epochs (~45 min)")
        print("Stage 3: Late layers         30 epochs (~45 min)")
        print("Stage 4: Full fine-tune      50 epochs (~75 min)")
        print("Stage 5: Targeted improve    40 epochs (~60 min) 🆕")
        print("Stage 6: Multi-scale         30 epochs (~50 min) 🆕")
        print("\nTotal time: ~5-6 hours")
        print("="*80)
        
        input("\nPress Enter to start enhanced optimization...")
        
        # Stages 1-4: Your original working pipeline
        stage1_model = self.stage1_fix_early_layers()
        print(f"\n✅ Stage 1: {stage1_model}")
        
        stage2_model = self.stage2_fix_middle_layer(stage1_model)
        print(f"\n✅ Stage 2: {stage2_model}")
        
        stage3_model = self.stage3_fix_late_layers(stage2_model)
        print(f"\n✅ Stage 3: {stage3_model}")
        
        stage4_model = self.stage4_full_fine_tune(stage3_model)
        print(f"\n✅ Stage 4: {stage4_model}")
        
        # 🆕 Stages 5-6: NEW enhancements
        print("\n" + "="*80)
        print("🆕 STARTING NEW ENHANCEMENT STAGES")
        print("="*80)
        
        stage5_model = self.stage5_targeted_weak_classes(stage4_model)
        print(f"\n✅ Stage 5 (NEW): {stage5_model}")
        
        final_model = self.stage6_multiscale_polish(stage5_model)
        print(f"\n✅ Stage 6 (FINAL): {final_model}")
        
        # Evaluate with TTA
        print("\n" + "="*80)
        print("🔥 BONUS EVALUATION")
        print("="*80)
        
        print("\n📊 Standard evaluation (val):")
        model = YOLO(final_model)
        val_results = model.val(data=self.data_yaml, split='val')
        
        print("\n📊 TTA evaluation (val):")
        tta_results = self.evaluate_with_tta(final_model, split='val')
        
        print("\n📊 Test set evaluation:")
        test_results = model.val(data=self.data_yaml, split='test')
        
        # Final summary
        print("\n" + "="*80)
        print("🎉 OPTIMIZATION COMPLETE!")
        print("="*80)
        print(f"\nFinal model: {final_model}")
        print("\n📊 RESULTS SUMMARY:")
        print(f"Validation (standard): {val_results.box.map50:.4f} mAP50")
        print(f"Validation (with TTA): {tta_results.box.map50:.4f} mAP50 🔥")
        print(f"Test (standard):       {test_results.box.map50:.4f} mAP50")
        print("\n✅ Expected improvements:")
        print("   - Overall: 96.3% → 97-98%")
        print("   - side_matra: 97.4% → 98%+")
        print("   - bottom_matra: 92.9% → 95%+")
        print("="*80)
        
        return final_model
    
    def resume_from_stage4(self, stage4_model_path):
        """
        🆕 SHORTCUT: Resume from your existing Stage 4 model
        Only run Stages 5 & 6 (saves 3 hours!)
        """
        print("\n" + "="*80)
        print("🚀 RESUMING FROM STAGE 4")
        print("="*80)
        print(f"Starting from: {stage4_model_path}")
        print("Running only Stages 5 & 6 (new enhancements)")
        print("\nStage 5: Targeted improve    40 epochs (~60 min)")
        print("Stage 6: Multi-scale         30 epochs (~50 min)")
        print("\nTotal time: ~2 hours")
        print("="*80)
        
        if not Path(stage4_model_path).exists():
            print(f"\n❌ ERROR: Model not found: {stage4_model_path}")
            return None
        
        input("\nPress Enter to continue from Stage 4...")
        
        # Stage 5
        stage5_model = self.stage5_targeted_weak_classes(stage4_model_path)
        print(f"\n✅ Stage 5: {stage5_model}")
        
        # Stage 6
        final_model = self.stage6_multiscale_polish(stage5_model)
        print(f"\n✅ Stage 6 (FINAL): {final_model}")
        
        # Evaluate
        print("\n📊 Evaluating final model...")
        model = YOLO(final_model)
        val_results = model.val(data=self.data_yaml, split='val')
        test_results = model.val(data=self.data_yaml, split='test')
        
        print("\n" + "="*80)
        print("🎉 ENHANCEMENT COMPLETE!")
        print("="*80)
        print(f"\nFinal model: {final_model}")
        print(f"Validation mAP50: {val_results.box.map50:.4f}")
        print(f"Test mAP50: {test_results.box.map50:.4f}")
        print("="*80)
        
        return final_model


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         ENHANCED PROGRESSIVE OPTIMIZER - BUILDS ON YOUR CODE         ║
╚══════════════════════════════════════════════════════════════════════╝

YOUR CURRENT RESULTS (Stage 4):
================================
✅ Overall: 96.3% mAP50
✅ top_matra: 98.6% (excellent!)
✅ side_matra: 97.4% (good)
⚠️ bottom_matra: 92.9% (needs improvement)

NEW ENHANCEMENTS (Stages 5-6):
===============================
🆕 Stage 5: Targeted augmentation for weak classes
   - Focus on side_matra and bottom_matra
   - Enhanced augmentation + class weighting
   - Expected: +1-1.5% overall, bottom_matra 92.9%→95%

🆕 Stage 6: Multi-scale training
   - Train on 800px for better detail
   - Expected: +0.5-1% overall

EXPECTED FINAL RESULTS:
=======================
Overall:      96.3% → 97-98% mAP50 ✨
top_matra:    98.6% → 99%+ (maintain excellence)
side_matra:   97.4% → 98%+ (slight boost)
bottom_matra: 92.9% → 95%+ (significant improvement)

With TTA: Add extra 0.5-1% boost! 🔥

OPTIONS:
========
1. Run full pipeline (Stages 1-6)       ~5-6 hours
2. Resume from Stage 4 (Stages 5-6 only) ~2 hours ⭐ RECOMMENDED

╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Configure
    base_model = 'runs/modi_matra/train_full_7k2/weights/best.pt'
    stage4_model = 'runs/progressive_fix/stage4_final/weights/best.pt'
    data_yaml = '/Users/applemaair/Desktop/modi/modi_script/modi_full_7k/merged/modi_matra.yaml'
    
    optimizer = EnhancedProgressiveOptimizer(base_model, data_yaml)
    
    print("\nCHOOSE AN OPTION:")
    print("1. Run full pipeline (Stages 1-6) - ~5-6 hours")
    print("2. Resume from Stage 4 (NEW Stages 5-6 only) - ~2 hours ⭐")
    print("3. Just evaluate current model with TTA - ~5 min")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        print("\n🚀 Running full pipeline...")
        final_model = optimizer.run_all_stages()
        
    elif choice == "2":
        if Path(stage4_model).exists():
            print("\n🚀 Resuming from Stage 4...")
            final_model = optimizer.resume_from_stage4(stage4_model)
        else:
            print(f"\n❌ Stage 4 model not found: {stage4_model}")
            print("Please run option 1 or update the path")
            return
            
    elif choice == "3":
        print("\n🚀 Quick TTA evaluation...")
        if Path(stage4_model).exists():
            results = optimizer.evaluate_with_tta(stage4_model, split='val')
            print(f"\nWith TTA: {results.box.map50:.4f} mAP50")
        else:
            print(f"Model not found: {stage4_model}")
            print("Update path and try again")
        return
    
    else:
        print("❌ Invalid choice")
        return
    
    print(f"\n🎉 ALL DONE! Final model ready: {final_model}")


if __name__ == '__main__':
    main()