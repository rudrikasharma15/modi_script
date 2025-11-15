# Modi Script Matra Segmentation Pipeline

Complete end-to-end pipeline for detecting and segmenting matras (vowel marks) in Modi script using YOLOv8.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Detailed Workflow](#detailed-workflow)
5. [Scripts Reference](#scripts-reference)
6. [Tips & Best Practices](#tips--best-practices)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This pipeline trains a YOLO object detection model to identify three types of matras in Modi script:
- **Top Matra** (class 0): Marks appearing above base characters (i, e, ai, anusvara)
- **Side Matra** (class 1): Marks appearing on the right side (aa, o, au, visarga)
- **Bottom Matra** (class 2): Marks appearing below (u, uu)

**Important**: Your dataset contains complete characters (base + matra). You need to manually annotate the matra regions within each character.

---

## 🚀 Installation

### Step 1: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv modi_env
source modi_env/bin/activate  # On Windows: modi_env\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
# Check YOLO installation
yolo version

# Check if CUDA is available (for GPU training)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## ⚡ Quick Start

### Complete Pipeline in 6 Steps

```bash
# 1. PREPARE DATASET (Start with 100 images sample)
python prepare_modi_dataset.py \
    --data_root "/Users/applemaair/Downloads/Dataset_Modi/Dataset_Modi" \
    --out datasets/modi_sample \
    --sample 100

# 2. ANNOTATE (Manual step - use LabelImg)
labelImg datasets/modi_sample/images datasets/modi_sample/labels
# Draw boxes around ONLY the matra parts, not the base character
# Press W to draw box, select class (0=top, 1=side, 2=bottom)
# Press Ctrl+S to save, D for next image

# 3. VISUALIZE YOUR ANNOTATIONS (optional but recommended)
python visualize_annotations.py \
    --dataset datasets/modi_sample \
    --sample 10

# 4. CREATE TRAIN/VAL SPLITS
python create_splits.py \
    --dataset datasets/modi_sample \
    --split 0.85

# 5. TRAIN MODEL
python train_modi_matra.py \
    --data datasets/modi_sample/modi_matra.yaml \
    --epochs 100 \
    --batch 16

# 6. TEST PREDICTIONS
python predict_modi_matra.py \
    --model runs/modi_matra/train/weights/best.pt \
    --source test_images/ \
    --conf 0.25
```

---

## 📚 Detailed Workflow

### Phase 1: Dataset Preparation

#### 1.1 Understand Your Data Structure

Your dataset should look like:
```
Dataset_Modi/
├── 1 a-ananas/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── 11 KA-kadhai/
├── 13 KI-kiran/
└── ...
```

#### 1.2 Prepare Sample Dataset

**Start small!** Begin with 100-300 images to validate your workflow.

```bash
# Prepare 100 images per matra type (300 total)
python prepare_modi_dataset.py \
    --data_root "/path/to/Dataset_Modi" \
    --out datasets/modi_sample \
    --sample 100
```

**Output:**
- `datasets/modi_sample/images/` - All images copied here
- `datasets/modi_sample/labels/` - Empty label files created
- `datasets/modi_sample/metadata.json` - Reference info
- `datasets/modi_sample/ANNOTATION_GUIDE.md` - Read this!

**For full dataset (later):**
```bash
python prepare_modi_dataset.py \
    --data_root "/path/to/Dataset_Modi" \
    --out datasets/modi_full
```

---

### Phase 2: Manual Annotation

This is the **most critical step**. Quality annotations = good model.

#### 2.1 Read the Annotation Guide

```bash
cat datasets/modi_sample/ANNOTATION_GUIDE.md
```

#### 2.2 Start LabelImg

```bash
labelImg datasets/modi_sample/images datasets/modi_sample/labels
```

#### 2.3 LabelImg Setup (First Time Only)

1. **Switch to YOLO format**: View → YOLO Format (or press Ctrl+Y)
2. **Enable auto-save**: View → Auto Save Mode ✓
3. **Load classes**: The tool will auto-load from `classes.txt`

#### 2.4 Annotation Process

For **each image**:

1. **Identify the matra** (check `metadata.json` for hints)
2. **Press W** to start drawing a bounding box
3. **Draw box ONLY around the matra part**
   - ❌ Don't include the base consonant
   - ✓ Keep box tight around matra strokes
4. **Select the correct class**:
   - `0` = top_matra (above character)
   - `1` = side_matra (right side)
   - `2` = bottom_matra (below)
5. **Press Ctrl+S** to save (or auto-saves)
6. **Press D** to go to next image

#### 2.5 Annotation Examples

```
Top Matra Example (Class 0):
┌─────────┐ ← Draw box around this horizontal line
│   ──    │
└─────────┘
    क
    ↑ Base character (don't include in box)

Side Matra Example (Class 1):
    क│  ← Draw box around this vertical mark
     │
     
Bottom Matra Example (Class 2):
    क
    ˘  ← Draw box around this mark below
```

#### 2.6 Annotation Tips

- **Quality over speed**: Good annotations take time
- **Be consistent**: Annotate similar matras the same way
- **Take breaks**: Annotation fatigue leads to mistakes
- **Review periodically**: Check your first 20-30 annotations
- **Use metadata.json**: It tells you the expected matra type

**Time estimates:**
- 100 images: 1-2 hours (with practice)
- 300 images: 3-5 hours
- 7000 images: 40-60 hours (do in batches!)

---

### Phase 3: Validation & Splitting

#### 3.1 Visualize Your Annotations

Before training, **verify your annotations**:

```bash
# View random sample of 20 annotated images
python visualize_annotations.py \
    --dataset datasets/modi_sample \
    --sample 20

# Or save visualizations to folder
python visualize_annotations.py \
    --dataset datasets/modi_sample \
    --output viz_check/
```

**Check for:**
- ✓ Boxes are around matras only (not whole characters)
- ✓ Class labels are correct (green=top, blue=side, red=bottom)
- ✓ No boxes overlap incorrectly
- ✓ Boxes are tight around strokes

#### 3.2 Create Train/Val Splits

```bash
python create_splits.py \
    --dataset datasets/modi_sample \
    --split 0.85  # 85% train, 15% validation
```

**Output:**
- `images/train/` and `labels/train/` - Training data
- `images/val/` and `labels/val/` - Validation data
- `modi_matra.yaml` - YOLO config file

---

### Phase 4: Training

#### 4.1 Basic Training

```bash
python train_modi_matra.py \
    --data datasets/modi_sample/modi_matra.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 640
```

**Training parameters explained:**
- `--epochs 100`: Number of complete passes through dataset
- `--batch 16`: Images processed simultaneously (adjust for GPU memory)
- `--imgsz 640`: Input image size (larger = slower but potentially better)

#### 4.2 Advanced Training Options

```bash
# Longer training with larger model
python train_modi_matra.py \
    --data datasets/modi_full/modi_matra.yaml \
    --model yolov8s.pt \
    --epochs 150 \
    --batch 32 \
    --imgsz 800 \
    --patience 30

# Resume interrupted training
python train_modi_matra.py \
    --data datasets/modi_sample/modi_matra.yaml \
    --model runs/modi_matra/train/weights/last.pt \
    --epochs 150
```

**Model sizes** (speed vs accuracy tradeoff):
- `yolov8n.pt` - Nano (fastest, least accurate) ← **Start here**
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra large (slowest, most accurate)

#### 4.3 Monitor Training

Training will save to `runs/modi_matra/train/`:

```bash
# View training curves
open runs/modi_matra/train/results.png

# Check metrics during training
tail -f runs/modi_matra/train/results.csv
```

**Key metrics to watch:**
- `mAP50` - Main accuracy metric (higher is better)
- `precision` - Percentage of correct detections
- `recall` - Percentage of matras found
- `box_loss` - Bounding box accuracy (lower is better)

#### 4.4 What to Expect

**Good signs:**
- ✓ mAP50 > 0.7 after 50-100 epochs
- ✓ Losses steadily decreasing
- ✓ Val metrics close to train metrics

**Bad signs:**
- ❌ mAP50 < 0.3 after 100 epochs → Need more/better annotations
- ❌ Train metrics good but val metrics poor → Overfitting
- ❌ Losses not decreasing → Try different learning rate

---

### Phase 5: Inference & Testing

#### 5.1 Single Image Prediction

```bash
python predict_modi_matra.py \
    --model runs/modi_matra/train/weights/best.pt \
    --source test_image.jpg \
    --conf 0.25
```

#### 5.2 Batch Prediction

```bash
# Process entire folder
python predict_modi_matra.py \
    --model runs/modi_matra/train/weights/best.pt \
    --source test_images/ \
    --conf 0.25 \
    --output my_predictions/
```

#### 5.3 Save Results as JSON

```bash
python predict_modi_matra.py \
    --model runs/modi_matra/train/weights/best.pt \
    --source test_images/ \
    --json
```

**Output JSON format:**
```json
[
  {
    "image": "test1.jpg",
    "predictions": [
      {
        "class": "top_matra",
        "class_id": 0,
        "confidence": 0.89,
        "bbox": [120, 45, 180, 75]
      }
    ]
  }
]
```

#### 5.4 Using Trained Model with YOLO CLI

```bash
# Direct YOLO prediction
yolo detect predict \
    model=runs/modi_matra/train/weights/best.pt \
    source=test_images/ \
    conf=0.25 \
    save=True
```

---

## 📖 Scripts Reference

### prepare_modi_dataset.py

Organizes your Modi character images for annotation.

**Arguments:**
- `--data_root` (required): Path to Dataset_Modi folder
- `--out`: Output directory (default: `datasets/modi_matra`)
- `--sample`: Sample N images per matra type (for testing)

**Example:**
```bash
python prepare_modi_dataset.py \
    --data_root ./Dataset_Modi \
    --out datasets/modi_sample \
    --sample 100
```

---

### create_splits.py

Creates train/val splits from annotated data.

**Arguments:**
- `--dataset` (required): Dataset directory from prepare script
- `--split`: Train ratio (default: 0.85 = 85% train, 15% val)

**Example:**
```bash
python create_splits.py \
    --dataset datasets/modi_sample \
    --split 0.85
```

---

### visualize_annotations.py

Visualizes annotated bounding boxes on images.

**Arguments:**
- `--dataset` (required): Dataset directory
- `--split`: Which split to view (`all`, `train`, `val`, `none`)
- `--output`: Save to folder instead of displaying
- `--sample`: Visualize N random images

**Examples:**
```bash
# Interactive viewing
python visualize_annotations.py --dataset datasets/modi_sample

# Save to folder
python visualize_annotations.py \
    --dataset datasets/modi_sample \
    --output visualizations/

# View only training set
python visualize_annotations.py \
    --dataset datasets/modi_sample \
    --split train
```

---

### train_modi_matra.py

Trains YOLOv8 model for matra detection.

**Arguments:**
- `--data` (required): Path to YAML config
- `--model`: YOLO model to use (default: `yolov8n.pt`)
- `--epochs`: Training epochs (default: 100)
- `--batch`: Batch size (default: 16)
- `--imgsz`: Image size (default: 640)
- `--patience`: Early stopping patience (default: 20)
- `--device`: Device (`cpu`, `0`, `0,1`, etc.)
- `--project`: Save directory (default: `runs/modi_matra`)
- `--name`: Experiment name (default: `train`)

**Example:**
```bash
python train_modi_matra.py \
    --data datasets/modi_sample/modi_matra.yaml \
    --model yolov8s.pt \
    --epochs 150 \
    --batch 32 \
    --device 0
```

---

### predict_modi_matra.py

Runs inference on new images.

**Arguments:**
- `--model` (required): Path to trained model
- `--source` (required): Image or folder to process
- `--conf`: Confidence threshold (default: 0.25)
- `--output`: Output directory (default: `predictions`)
- `--json`: Save predictions as JSON

**Example:**
```bash
python predict_modi_matra.py \
    --model runs/modi_matra/train/weights/best.pt \
    --source test_images/ \
    --conf 0.3 \
    --json
```

---

## 💡 Tips & Best Practices

### Data Quality

1. **Start small**: Annotate 100-300 images first
2. **Quality matters**: 100 perfect annotations > 1000 sloppy ones
3. **Be consistent**: Use the same annotation style throughout
4. **Review early**: Check your first 20-30 annotations carefully

### Annotation Best Practices

- ✓ Draw boxes **tight** around matra strokes
- ✓ Exclude base consonants completely
- ✓ For split matras (top+side), create separate boxes
- ✓ Use `metadata.json` as reference
- ❌ Don't guess - if unsure, skip the image
- ❌ Don't rush - take breaks to maintain quality

### Training Tips

1. **Use GPU if available**: 10-50x faster than CPU
2. **Start with small model**: `yolov8n.pt` trains faster
3. **Monitor overfitting**: If val_mAP << train_mAP, you're overfitting
4. **Try data augmentation**: Enabled by default in training script
5. **Patience parameter**: Stops training if no improvement for N epochs

### Performance Optimization

- **Batch size**: Increase if you have GPU memory (`--batch 32`)
- **Image size**: Larger = slower but may improve accuracy (`--imgsz 800`)
- **Model size**: Upgrade to `yolov8s.pt` or `yolov8m.pt` for better accuracy
- **More data**: More annotations generally = better performance

### Iterative Improvement

1. Train on 300 images
2. Test predictions on validation set
3. Identify failure cases
4. Add more annotations for difficult cases
5. Retrain
6. Repeat until satisfied

---

## 🔧 Troubleshooting

### Issue: "No annotated images found"

**Cause**: Label files are empty (no annotations)

**Solution:**
```bash
# Check if you have annotations
python create_splits.py --dataset datasets/modi_sample
# It will show count of annotated vs empty labels
```

---

### Issue: "CUDA out of memory"

**Cause**: Batch size too large for GPU

**Solution:**
```bash
# Reduce batch size
python train_modi_matra.py \
    --data datasets/modi_sample/modi_matra.yaml \
    --batch 8  # Try 8, 4, or even 1
```

---

### Issue: Training is very slow

**Solutions:**
1. Use GPU instead of CPU (100x faster)
2. Reduce `--imgsz` from 640 to 512 or 416
3. Use smaller model (`yolov8n.pt`)
4. Reduce `--batch` size if GPU memory limited

---

### Issue: Low mAP (< 0.5) after training

**Possible causes:**

1. **Not enough data**: Annotate more images (aim for 500+ minimum)
2. **Poor annotations**: Review and fix annotation quality
3. **Imbalanced classes**: Check if one matra type dominates
4. **Model too small**: Try `yolov8s.pt` instead of `yolov8n.pt`
5. **Too few epochs**: Train longer (150-200 epochs)

**Debug:**
```bash
# Visualize some predictions
python predict_modi_matra.py \
    --model runs/modi_matra/train/weights/best.pt \
    --source datasets/modi_sample/images/val/ \
    --conf 0.15  # Lower confidence to see all predictions
```

---

### Issue: Model detects whole characters, not just matras

**Cause**: Annotations include base characters

**Solution:**
- Re-check your annotations using `visualize_annotations.py`
- Boxes should be around **matras only**, not complete characters
- Re-annotate if needed, focusing on matra regions

---

### Issue: "Dataset YAML not found"

**Cause**: Haven't run `create_splits.py` yet

**Solution:**
```bash
# Must create splits before training
python create_splits.py --dataset datasets/modi_sample
```

---

### Issue: LabelImg not opening

**Solution:**
```bash
# Reinstall LabelImg
pip uninstall labelImg
pip install labelImg

# If still issues, try system-specific install
# On Ubuntu/Debian:
sudo apt-get install labelimg

# On macOS:
brew install labelimg
```

---

## 📊 Expected Timeline

**For 300 images (recommended start):**
- Dataset preparation: 10 minutes
- Annotation: 3-5 hours
- Training: 30-60 minutes (with GPU)
- Testing & iteration: 1-2 hours
- **Total: 5-9 hours**

**For 7000 images (full dataset):**
- Annotation: 40-60 hours (spread over days/weeks)
- Training: 2-4 hours (with GPU)
- **Total: 45-65 hours**

