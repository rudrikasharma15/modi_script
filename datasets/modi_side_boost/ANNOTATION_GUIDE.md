# Modi Matra Annotation Guide

## Dataset Information
Total images prepared: 100
- Top matras: 0
- Side matras: 100
- Bottom matras: 0

Target matra types: side_matra
Skipped folders (no matras): 243

## Classes (YOLO Format)
0 - top_matra (appears above base character)
1 - side_matra (appears to the right/side of base)
2 - bottom_matra (appears below base character)

## ⚠️ IMPORTANT ANNOTATION RULES

### Rule 1: Annotate ALL Matras You See! ✅

**Even if you're focusing on side matras, if an image has BOTH side + top matras:**
- Draw TWO boxes (one for each matra)
- Label each correctly based on position

**Example - KAO character (compound vowel):**
```
Image shows: Base + top mark (o component) + side mark (aa)

Your annotation:
✅ Box 1: Around top mark → Label: top_matra (class 0)
✅ Box 2: Around side mark → Label: side_matra (class 1)

❌ WRONG: Only annotating the side matra
```

### Rule 2: Label by POSITION, Not Folder Name

The folder name is just a hint. Trust your eyes:
- Mark ABOVE base → top_matra (0)
- Mark on RIGHT side → side_matra (1)  
- Mark BELOW base → bottom_matra (2)

### Rule 3: Keep Boxes Small (10-25% of image width)

✅ CORRECT: Tight box around matra only
❌ WRONG: Box covering whole character
❌ WRONG: Box including base consonant

## Setup LabelImg

1. Install LabelImg:
   ```
   pip install labelImg
   ```

2. Start annotation:
   ```
   labelImg /Users/applemaair/Desktop/modi/modi_script/datasets/modi_side_boost/images /Users/applemaair/Desktop/modi/modi_script/datasets/modi_side_boost/labels
   ```

3. In LabelImg settings:
   - View > Auto Save Mode (recommended)
   - Press Ctrl+Y to switch to YOLO format
   - File > Change Save Dir > Select: /Users/applemaair/Desktop/modi/modi_script/datasets/modi_side_boost/labels

## Annotation Workflow

1. **Look at the image** - Identify ALL matras present
2. **Press W** - Start drawing first box
3. **Draw tight box** around first matra (10-25% width)
4. **Select class** based on POSITION (0=top, 1=side, 2=bottom)
5. **If multiple matras exist:**
   - Press W again
   - Draw second box around second matra
   - Select its class
6. **Press D** - Next image

## Common Scenarios

### Scenario A: Single Matra Image ✅
```
Image: KAA_image.png
Contains: Base + side matra only

Action: Draw 1 box around side matra
Label: side_matra (class 1)
```

### Scenario B: Compound Matra Image ✅✅
```
Image: KAO_image.png  
Contains: Base + top mark + side mark

Action: Draw 2 boxes
Box 1: Around top mark → top_matra (0)
Box 2: Around side mark → side_matra (1)

⚠️ Don't skip the top matra just because 
   this is from a "side matra" folder!
```

### Scenario C: Triple Matra Image ✅✅✅
```
Image: Complex compound character
Contains: Top + side + bottom marks

Action: Draw 3 boxes, label each correctly
```

## Matra Position Reference

**Top Matras (Class 0):**
- i (ि) - Short vertical stroke above
- e (े) - Horizontal line above  
- ai (ै) - Two marks above
- nm/am (ं) - Dot/circle above

**Side Matras (Class 1):**
- aa (ा) - Vertical line to the right
- o (ो) - Curved mark on right
- au (ौ) - Additional marks on right
- ahaa (ः) - Two dots on right

**Bottom Matras (Class 2):**
- u (ु) - Small mark below
- uu (ू) - Longer mark below

## Quality Checks

Before finishing, verify:
- [ ] ALL matras annotated (even if not the "target" type)
- [ ] Boxes are tight (10-25% width, around matras only)
- [ ] Class labels correct (0=top, 1=side, 2=bottom based on position)
- [ ] No matras missed (especially in compound characters)
- [ ] Consistent box sizing for similar matras

## After Annotation

Once you've annotated all images:

1. Run the split creation script:
   ```
   python create_splits.py --dataset /Users/applemaair/Desktop/modi/modi_script/datasets/modi_side_boost
   ```

2. Train the model:
   ```
   python train_modi_matra.py --data /Users/applemaair/Desktop/modi/modi_script/datasets/modi_side_boost/modi_matra.yaml --epochs 150 --batch 16
   ```

## Keyboard Shortcuts in LabelImg

- W - Create bounding box
- D - Next image
- A - Previous image  
- Del - Delete selected box
- Ctrl+S - Save
- Ctrl+D - Duplicate box

## Tips for Success

✓ **Annotate ALL matras** - Even if focusing on one type
✓ **Tight boxes** - 10-25% of image width maximum
✓ **Trust position** - Not folder name
✓ **Be consistent** - Similar matras = similar boxes
✓ **Take breaks** - Every 20 images, check your quality

## Need Help?

- Check metadata.json for expected primary matra type
- Your model handles multiple matras per image automatically
- When in doubt, annotate everything you see!
