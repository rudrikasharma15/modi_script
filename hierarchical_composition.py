"""
Hierarchical Composition Post-Processing
Combines detected characters + matras into final graphemes
"""

import numpy as np
from scipy.spatial.distance import cdist

class HierarchicalComposer:
    """
    Composes character+matra detections into complete graphemes
    Uses spatial proximity to associate matras with characters
    """
    
    def __init__(self, 
                 char_classes=list(range(34)),  # Classes 0-33
                 matra_classes=list(range(34, 37)),  # Classes 34-36
                 proximity_threshold=0.3):
        
        self.char_classes = char_classes
        self.matra_classes = matra_classes
        self.proximity_threshold = proximity_threshold
        
        # Character names
        self.char_names = [
            'ka', 'kha', 'ga', 'gha', 'nga',
            'cha', 'chha', 'ja', 'jha', 'nya',
            'ta', 'tha', 'da', 'dha', 'na',
            'taa', 'thaa', 'daa', 'dhaa', 'naa',
            'pa', 'pha', 'ba', 'bha', 'ma',
            'ya', 'ra', 'la', 'va', 'sha',
            'shha', 'sa', 'ha', 'ksha'
        ]
        
        # Matra names
        self.matra_names = ['top_matra', 'side_matra', 'bottom_matra']
    
    def compute_spatial_distance(self, char_box, matra_box):
        """
        Compute spatial distance between character and matra centers
        Uses normalized Euclidean distance
        
        Args:
            char_box: [x_center, y_center, width, height]
            matra_box: [x_center, y_center, width, height]
        
        Returns:
            distance: Float between 0-1
        """
        char_center = np.array([char_box[0], char_box[1]])
        matra_center = np.array([matra_box[0], matra_box[1]])
        
        return np.linalg.norm(char_center - matra_center)
    
    def associate_matras_with_characters(self, detections):
        """
        Associate each matra with nearest character
        
        Args:
            detections: List of dicts with keys:
                - class: int (0-36)
                - bbox: [x_center, y_center, width, height]
                - conf: float
        
        Returns:
            compositions: List of dicts:
                - character: {class, bbox, conf, name}
                - matras: [{type, bbox, conf}, ...]
                - composed_name: str (e.g., "ka + top_matra")
        """
        
        # Separate characters and matras
        characters = [d for d in detections if d['class'] in self.char_classes]
        matras = [d for d in detections if d['class'] in self.matra_classes]
        
        if len(characters) == 0:
            return []
        
        compositions = []
        
        for char in characters:
            char_box = char['bbox']
            
            # Find matras within proximity threshold
            associated_matras = []
            
            for matra in matras:
                matra_box = matra['bbox']
                distance = self.compute_spatial_distance(char_box, matra_box)
                
                if distance <= self.proximity_threshold:
                    matra_type = self.matra_names[matra['class'] - 34]
                    associated_matras.append({
                        'type': matra_type,
                        'class': matra['class'],
                        'bbox': matra_box,
                        'conf': matra['conf'],
                        'distance': distance
                    })
            
            # Sort matras by distance (closest first)
            associated_matras.sort(key=lambda x: x['distance'])
            
            # Create composition
            char_name = self.char_names[char['class']]
            
            if len(associated_matras) > 0:
                matra_str = " + ".join([m['type'] for m in associated_matras])
                composed_name = f"{char_name} + {matra_str}"
            else:
                composed_name = char_name
            
            compositions.append({
                'character': {
                    'class': char['class'],
                    'name': char_name,
                    'bbox': char_box,
                    'conf': char['conf']
                },
                'matras': associated_matras,
                'composed_name': composed_name,
                'confidence': min(char['conf'], 
                                min([m['conf'] for m in associated_matras], default=1.0))
            })
        
        return compositions
    
    def compute_composition_accuracy(self, predictions, ground_truth):
        """
        Compute accuracy of composed graphemes
        
        Args:
            predictions: List of compositions from associate_matras_with_characters
            ground_truth: List of ground truth compositions
        
        Returns:
            metrics: Dict with accuracy, precision, recall, f1
        """
        
        total_chars = len(ground_truth)
        correct_chars = 0
        correct_matras = 0
        total_matras = 0
        
        for pred_comp in predictions:
            pred_char = pred_comp['character']['class']
            pred_matras = set([m['class'] for m in pred_comp['matras']])
            
            # Find matching ground truth
            matching_gt = None
            for gt_comp in ground_truth:
                if self._boxes_overlap(pred_comp['character']['bbox'], 
                                      gt_comp['character']['bbox']):
                    matching_gt = gt_comp
                    break
            
            if matching_gt:
                # Check character accuracy
                if pred_char == matching_gt['character']['class']:
                    correct_chars += 1
                
                # Check matra accuracy
                gt_matras = set([m['class'] for m in matching_gt['matras']])
                correct_matras += len(pred_matras & gt_matras)
                total_matras += len(gt_matras)
        
        char_accuracy = 100 * correct_chars / total_chars if total_chars > 0 else 0
        matra_accuracy = 100 * correct_matras / total_matras if total_matras > 0 else 0
        
        # Combined accuracy (both character AND matras must be correct)
        correct_compositions = 0
        for pred_comp in predictions:
            for gt_comp in ground_truth:
                if self._compositions_match(pred_comp, gt_comp):
                    correct_compositions += 1
                    break
        
        composition_accuracy = 100 * correct_compositions / total_chars if total_chars > 0 else 0
        
        return {
            'character_accuracy': char_accuracy,
            'matra_accuracy': matra_accuracy,
            'composition_accuracy': composition_accuracy,
            'total_characters': total_chars,
            'correct_characters': correct_chars,
            'correct_matras': correct_matras,
            'total_matras': total_matras
        }
    
    def _boxes_overlap(self, box1, box2, iou_threshold=0.5):
        """Check if two boxes overlap with IoU > threshold"""
        x1_min = box1[0] - box1[2]/2
        x1_max = box1[0] + box1[2]/2
        y1_min = box1[1] - box1[3]/2
        y1_max = box1[1] + box1[3]/2
        
        x2_min = box2[0] - box2[2]/2
        x2_max = box2[0] + box2[2]/2
        y2_min = box2[1] - box2[3]/2
        y2_max = box2[1] + box2[3]/2
        
        # Intersection
        x_inter_min = max(x1_min, x2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_min = max(y1_min, y2_min)
        y_inter_max = min(y1_max, y2_max)
        
        if x_inter_max <= x_inter_min or y_inter_max <= y_inter_min:
            return False
        
        inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / union_area if union_area > 0 else 0
        
        return iou >= iou_threshold
    
    def _compositions_match(self, pred_comp, gt_comp):
        """Check if predicted and ground truth compositions match"""
        # Character must match
        if pred_comp['character']['class'] != gt_comp['character']['class']:
            return False
        
        # Matras must match (set equality)
        pred_matras = set([m['class'] for m in pred_comp['matras']])
        gt_matras = set([m['class'] for m in gt_comp['matras']])
        
        return pred_matras == gt_matras
    
    def visualize_composition(self, image, compositions, save_path=None):
        """
        Visualize composed graphemes on image
        """
        import cv2
        
        img = image.copy()
        h, w = img.shape[:2]
        
        for comp in compositions:
            # Draw character box (green)
            char_box = comp['character']['bbox']
            x1 = int((char_box[0] - char_box[2]/2) * w)
            y1 = int((char_box[1] - char_box[3]/2) * h)
            x2 = int((char_box[0] + char_box[2]/2) * w)
            y2 = int((char_box[1] + char_box[3]/2) * h)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw character name
            cv2.putText(img, comp['character']['name'], 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 2)
            
            # Draw matra boxes (different colors)
            colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0)]  # Blue, Red, Yellow
            
            for i, matra in enumerate(comp['matras']):
                m_box = matra['bbox']
                mx1 = int((m_box[0] - m_box[2]/2) * w)
                my1 = int((m_box[1] - m_box[3]/2) * h)
                mx2 = int((m_box[0] + m_box[2]/2) * w)
                my2 = int((m_box[1] + m_box[3]/2) * h)
                
                color = colors[i % len(colors)]
                cv2.rectangle(img, (mx1, my1), (mx2, my2), color, 2)
                
                # Draw matra type
                cv2.putText(img, matra['type'], 
                           (mx1, my1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, color, 1)
            
            # Draw composed name at bottom
            cv2.putText(img, comp['composed_name'], 
                       (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, img)
        
        return img


# USAGE EXAMPLE
if __name__ == "__main__":
    from ultralytics import YOLO
    import cv2
    
    # Load trained model
    model = YOLO('runs/hierarchical_yolo/char_matra_detection/weights/best.pt')
    
    # Initialize composer
    composer = HierarchicalComposer(proximity_threshold=0.3)
    
    # Run inference on test image
    results = model('path/to/test/image.jpg')
    
    # Extract detections
    detections = []
    for box in results[0].boxes:
        detections.append({
            'class': int(box.cls[0]),
            'bbox': box.xywhn[0].tolist(),  # Normalized xywh
            'conf': float(box.conf[0])
        })
    
    # Compose graphemes
    compositions = composer.associate_matras_with_characters(detections)
    
    # Print results
    print(f"\n🎯 Detected {len(compositions)} composed graphemes:")
    for comp in compositions:
        print(f"  - {comp['composed_name']} (conf: {comp['confidence']:.2f})")
    
    # Visualize
    img = cv2.imread('path/to/test/image.jpg')
    viz_img = composer.visualize_composition(img, compositions, 'output_composition.jpg')
    
    print("\n✅ Visualization saved to output_composition.jpg")