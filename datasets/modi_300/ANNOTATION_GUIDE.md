# Modi Matra Annotation Guide

## Dataset Information
Total images prepared: 300
- Top matras: 100
- Side matras: 100
- Bottom matras: 100

Skipped folders (no matras): 243

## Classes (YOLO Format)
0 - top_matra (appears above base character)
1 - side_matra (appears to the right/side of base)
2 - bottom_matra (appears below base character)

## Setup LabelImg

1. Install LabelImg:
   ```
   pip install labelImg
   ```

2. Start annotation:
   ```
   labelImg /Users/applemaair/Desktop/modi/modi_script/datasets/modi_300/images /Users/applemaair/Desktop/modi/modi_script/datasets/modi_300/labels
   ```

3. In LabelImg settings:
   - View > Auto Save Mode (recommended)
   - Press Ctrl+Y to switch to YOLO format
   - File > Change Save Dir > Select: /Users/applemaair/Desktop/modi/modi_script/datasets/modi_300/labels

## Annotation Instructions

### How to Annotate Each Image

1. **Understand what you see**: Each image shows a complete Modi character (base + matra)

2. **Identify the matra**: Look for the vowel mark that modifies the base consonant
   - Check metadata.json for hints about expected matra type
   
3. **Draw bounding box**: Press 'W' key
   - Draw box ONLY around the matra part
   - Do NOT include the base consonant in the box
   - Keep box tight around the matra strokes

4. **Select class**: Choose the correct class number
   - 0 = top_matra (marks above)
   - 1 = side_matra (marks on right side)
   - 2 = bottom_matra (marks below)

5. **Save and continue**:
   - Press Ctrl+S to save (or auto-save if enabled)
   - Press D to go to next image

### Matra Examples

**Top Matras (Class 0):**
- i (ि) - Short vertical stroke above
- e (े) - Horizontal line above
- ai (ै) - Two marks above
- nm/am (ं) - Dot/circle above (anusvara)

**Side Matras (Class 1):**
- aa (ा) - Vertical line to the right
- o (ो) - Curved mark on right
- au (ौ) - Additional marks on right
- ahaa (ः) - Two dots on right (visarga)

**Bottom Matras (Class 2):**
- u (ु) - Small mark below
- uu (ू) - Longer mark below

## Annotation Tips

✓ **Focus only on matras** - Ignore the base consonant completely
✓ **Tight boxes** - Boxes should closely fit the matra strokes
✓ **Multiple parts** - If a matra has separate parts (e.g., top + side), create separate boxes
✓ **Connected writing** - For flowing/cursive Modi, use your best judgment for boundaries
✓ **Consistency** - Try to annotate similar matras the same way across images
✓ **When in doubt** - Check the metadata.json for the expected matra type

## Quality Checks

Before finishing, verify:
- [ ] All label files have content (not empty)
- [ ] Boxes are around matras, not whole characters
- [ ] Class labels are correct (0=top, 1=side, 2=bottom)
- [ ] No boxes overlap incorrectly
- [ ] Similar matras have similar annotations

## After Annotation

Once you've annotated all images:

1. Run the split creation script:
   ```
   python create_splits.py --dataset /Users/applemaair/Desktop/modi/modi_script/datasets/modi_300
   ```

2. Train the model:
   ```
   yolo detect train data=/Users/applemaair/Desktop/modi/modi_script/datasets/modi_300/modi_matra.yaml model=yolov8n.pt epochs=100
   ```

## Keyboard Shortcuts in LabelImg

- W - Create bounding box
- D - Next image
- A - Previous image
- Del - Delete selected box
- Ctrl+S - Save
- Ctrl+D - Duplicate box
- Space - Flag image as verified

## Need Help?

- Refer to metadata.json to see expected matra type for each image
- Start with a few images to establish your annotation style
- Take breaks to maintain consistency
- Review your first 20-30 annotations before continuing
