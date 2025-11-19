import cv2
import numpy as np
import os
from pathlib import Path

def preprocess_image(image):
    """Preprocess image for better character detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    return binary

def remove_lines(binary_image):
    """Remove grid lines while preserving characters"""
    # Create kernels for horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    
    # Detect horizontal lines
    horizontal_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, horizontal_kernel)
    # Detect vertical lines
    vertical_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, vertical_kernel)
    
    # Combine lines
    lines = cv2.add(horizontal_lines, vertical_lines)
    
    # Remove lines from original image
    result = cv2.subtract(binary_image, lines)
    
    return result

def find_and_crop_characters(image_path, output_dir, image_name):
    """Find and crop individual characters from the image"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return 0
    
    # Preprocess
    binary = preprocess_image(image)
    
    # Remove grid lines
    no_lines = remove_lines(binary)
    
    # Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(no_lines, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and sort contours
    valid_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Filter based on size (adjust these values based on your images)
        if area > 100 and w > 10 and h > 10 and w < 200 and h < 200:
            valid_contours.append((x, y, w, h))
    
    # Sort contours from top to bottom, left to right
    valid_contours = sorted(valid_contours, key=lambda c: (c[1] // 50, c[0]))
    
    # Crop and save each character
    saved_count = 0
    for idx, (x, y, w, h) in enumerate(valid_contours):
        # Add padding around the character
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        # Crop from original image
        cropped = image[y1:y2, x1:x2]
        
        # Skip if crop is too small
        if cropped.shape[0] < 20 or cropped.shape[1] < 20:
            continue
        
        # Save cropped character
        output_path = os.path.join(output_dir, f"{image_name}_char_{saved_count:03d}.png")
        cv2.imwrite(output_path, cropped)
        saved_count += 1
    
    return saved_count

def main():
    # Set up paths
    output_dir = "/Users/applemaair/Desktop/modi/modi_script/datasets/handwritten_images"
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Devanagari Character Cropping Tool")
    print("=" * 60)
    
    # Get input images
    print("\nPlease provide the paths to your images.")
    print("Enter image paths one by one (or 'done' when finished):")
    
    image_paths = []
    while True:
        path = input(f"Image {len(image_paths) + 1} path (or 'done'): ").strip()
        if path.lower() == 'done':
            break
        if os.path.exists(path):
            image_paths.append(path)
            print(f"✓ Added: {path}")
        else:
            print(f"✗ File not found: {path}")
    
    if not image_paths:
        print("\nNo valid images provided. Exiting.")
        return
    
    # Process each image
    total_chars = 0
    print(f"\n{'=' * 60}")
    print("Processing images...")
    print("=" * 60)
    
    for idx, image_path in enumerate(image_paths, 1):
        image_name = f"image_{idx}"
        print(f"\nProcessing {image_name}: {os.path.basename(image_path)}")
        
        char_count = find_and_crop_characters(image_path, output_dir, image_name)
        total_chars += char_count
        
        print(f"  → Extracted {char_count} characters")
    
    print(f"\n{'=' * 60}")
    print(f"✓ Complete! Total characters extracted: {total_chars}")
    print(f"✓ Saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()