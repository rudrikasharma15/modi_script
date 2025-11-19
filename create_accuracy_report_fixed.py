# create_accuracy_report_fixed.py
from pathlib import Path
import json

def analyze_predictions(pred_dir, image_dir):
    """Analyze predictions and create report"""
    
    pred_labels = Path(pred_dir)
    images = Path(image_dir)
    
    results = {
        'total_images': 0,
        'images_with_detections': 0,
        'total_detections': 0,
        'detections_by_class': {
            'top_matra': 0,
            'side_matra': 0,
            'bottom_matra': 0
        },
        'images_no_detection': [],
        'images_with_detection': []
    }
    
    class_names = ['top_matra', 'side_matra', 'bottom_matra']
    
    # Loop through all images (.jpg/.jpeg/.png)
    for img_path in sorted(images.glob('*.*')):
        if img_path.suffix.lower() not in ['.jpg', '.png', '.jpeg']:
            continue

        results['total_images'] += 1
        
        # TXT file must match EXACT image filename
        txt_path = pred_labels / f"{img_path.stem}.txt"
        
        if txt_path.exists():
            content = txt_path.read_text().strip()
            
            if content:
                # Detection found
                results['images_with_detections'] += 1
                results['images_with_detection'].append(img_path.name)
                
                detections = content.split("\n")
                results['total_detections'] += len(detections)
                
                for line in detections:
                    parts = line.split()
                    if not parts:
                        continue
                    class_id = int(parts[0])
                    results['detections_by_class'][class_names[class_id]] += 1
            else:
                # TXT exists but is empty
                results['images_no_detection'].append(img_path.name)
        else:
            # TXT file NOT found
            results['images_no_detection'].append(img_path.name)
    
    return results


# ----------------------------------------------------
# 🔥 USE THE CORRECT FOLDER FOR BOTH IMAGES AND LABELS
# ----------------------------------------------------
results = analyze_predictions(
    '/Users/applemaair/Desktop/modi/modi_script/test_predictions/handwritten_with_labels2',
    '/Users/applemaair/Desktop/modi/modi_script/test_predictions/handwritten_with_labels2'
)

# ----------------------------------------------------
# Print summary
# ----------------------------------------------------
print("\n" + "="*70)
print("📊 HANDWRITTEN MATRA TEST RESULTS")
print("="*70)
print(f"\n📁 Dataset:")
print(f"   Total images: {results['total_images']}")
print(f"   Images with detections: {results['images_with_detections']}")
print(f"   Images with no detection: {len(results['images_no_detection'])}")

detection_rate = (results['images_with_detections'] / results['total_images'] * 100) if results['total_images'] > 0 else 0
print(f"\n✅ Detection Rate: {detection_rate:.1f}%")

print(f"\n🎯 Total Detections: {results['total_detections']}")

if results['total_detections'] > 0:
    print("\n📈 Detections by Class:")
    for cls, count in results['detections_by_class'].items():
        percent = (count / results['total_detections']) * 100 if results['total_detections'] > 0 else 0
        print(f"   {cls:15s}: {count:3d} ({percent:.1f}%)")
else:
    print("\n❌ No detections found!")

if results['images_no_detection']:
    print(f"\n❌ Images with NO detection ({len(results['images_no_detection'])}):")
    for img in results['images_no_detection']:
        print(f"   - {img}")

print("\n" + "="*70)

# ----------------------------------------------------
# Save report JSON
# ----------------------------------------------------
report = {
    'summary': {
        'total_images': results['total_images'],
        'detection_rate': f"{detection_rate:.1f}%",
        'total_detections': results['total_detections']
    },
    'per_class': results['detections_by_class'],
    'failed_images': results['images_no_detection'],
    'successful_images': results['images_with_detection']
}

with open('handwritten_accuracy_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print("📄 Detailed report saved: handwritten_accuracy_report.json")
print("="*70 + "\n")
