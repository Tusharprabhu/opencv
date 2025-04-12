import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def detect_iris_in_eye(image_path, debug=True):
    # Read image and create eye detector
    img = cv2.imread(image_path)
    
    # Print image dimensions for debugging
    if debug:
        print(f"Image dimensions: {img.shape}")
    
    # Make a copy for visualization
    vis_img = img.copy()
    
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Convert to grayscale for eye detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    
    # Apply GaussianBlur to reduce noise and enhance edges
    gray_eq = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    
    # Use stricter parameters to reduce false positives
    eyes = eye_cascade.detectMultiScale(
        gray_eq,
        scaleFactor=1.05,
        minNeighbors=6,
        minSize=(30, 30)
    )
    
    if debug:
        print(f"Number of eyes detected: {len(eyes)}")
    
    # If no eyes detected with strict parameters, try with more relaxed parameters
    if len(eyes) == 0:
        eyes = eye_cascade.detectMultiScale(
            gray_eq,
            scaleFactor=1.03,
            minNeighbors=4,
            minSize=(25, 25)
        )
        if debug:
            print(f"Second attempt - Number of eyes detected: {len(eyes)}")
    
    # Filter eyes based on aspect ratio (real eyes are roughly circular/oval)
    filtered_eyes = []
    for (x, y, w, h) in eyes:
        aspect_ratio = float(w) / h
        if 0.7 <= aspect_ratio <= 1.6:
            filtered_eyes.append((x, y, w, h))
    
    if debug:
        print(f"Number of eyes after filtering: {len(filtered_eyes)}")
    
    # If no valid eyes after filtering, return None
    if len(filtered_eyes) == 0:
        print("No valid eye regions detected!")
        return None
    
    # Try each eye region to find iris
    all_results = []
    
    for eye_idx, (x, y, w, h) in enumerate(filtered_eyes):
        if debug:
            print(f"\nProcessing eye region {eye_idx+1}/{len(filtered_eyes)}")
        
        # Extract eye region
        eye_roi = img[y:y+h, x:x+w]
        original = eye_roi.copy()
        
        # Process the eye region for iris detection
        gray_roi = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        equalized = clahe.apply(gray_roi)
        
        # Calculate average intensity in the eye region
        avg_intensity = np.mean(gray_roi)
        if debug:
            print(f"Average eye intensity: {avg_intensity:.1f}")
        
        # Adjust threshold based on average intensity
        base_threshold = max(30, min(90, avg_intensity * 0.8))
        
        # Try multiple threshold values around the base threshold
        threshold_values = [base_threshold - 20, base_threshold, base_threshold + 20]
        best_contour = None
        best_score = 0
        best_threshold = 0
        best_thresh_img = None
        
        # Height and width of eye ROI for reference
        eye_height, eye_width = gray_roi.shape
        expected_radius = min(eye_width, eye_height) * 0.3  # Expected iris radius
        
        for threshold in threshold_values:
            if debug:
                print(f"Trying threshold: {threshold}")
            
            # Apply thresholding to isolate darker regions (iris)
            _, thresh = cv2.threshold(equalized, threshold, 255, cv2.THRESH_BINARY_INV)
            
            # Clean up the threshold image
            kernel = np.ones((3,3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process each contour
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Skip very small contours
                if area < 50:
                    continue
                    
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    # Calculate circularity (perfect circle = 1.0)
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Get the center and radius of a circle that encloses the contour
                    (cx, cy), radius = cv2.minEnclosingCircle(contour)
                    
                    # Skip if radius is too small or too large
                    if radius < 5 or radius > min(eye_width, eye_height) * 0.7:
                        continue
                    
                    # Calculate position score - iris should be near the center of the eye
                    center_x, center_y = eye_width / 2, eye_height / 2
                    distance_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                    max_distance = np.sqrt(center_x**2 + center_y**2)
                    position_score = 1.0 - (distance_from_center / max_distance)
                    
                    # Calculate radius score - how close is the radius to expected value
                    radius_diff = abs(radius - expected_radius)
                    radius_score = 1.0 - min(radius_diff / expected_radius, 1.0)
                    
                    # Calculate total score based on circularity, area, position and radius
                    if 0.5 < circularity < 1.0:
                        score = circularity * area * position_score * radius_score
                        if score > best_score:
                            best_score = score
                            best_contour = contour
                            best_threshold = threshold
                            best_thresh_img = thresh.copy()
                            if debug:
                                print(f"New best contour: score={score:.1f}, circularity={circularity:.2f}, radius={radius:.1f}")
        
        if best_contour is not None:
            # Fit circle to iris contour
            (ix, iy), radius = cv2.minEnclosingCircle(best_contour)
            center = (int(ix), int(iy))
            radius = int(radius)
            
            # Create visualization images
            iris_mask = np.zeros_like(gray_roi)
            cv2.circle(iris_mask, center, radius, 255, -1)
            
            # Extract iris region
            iris_region = cv2.bitwise_and(eye_roi, eye_roi, mask=iris_mask)
            
            # Draw iris circle on original eye image
            result = original.copy()
            cv2.circle(result, center, radius, (0, 255, 0), 2)
            
            # Draw iris center
            cv2.circle(result, center, 2, (0, 0, 255), -1)
            
            # Draw on the full image visualization copy
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.circle(vis_img, (x + int(ix), y + int(iy)), radius, (0, 255, 0), 2)
            cv2.circle(vis_img, (x + int(ix), y + int(iy)), 2, (0, 0, 255), -1)
            
            # Prepare result images for this eye
            result_images = [
                original,              # Original eye region
                equalized,             # Enhanced image
                best_thresh_img,       # Best threshold result
                iris_mask,             # Iris mask
                iris_region,           # Extracted iris
                result                 # Final result with circle
            ]
            
            all_results.append((result_images, best_score, (x, y, w, h), (ix, iy, radius)))
    
    # If we found any valid iris
    if all_results:
        # Sort results by score (descending)
        all_results.sort(key=lambda x: x[1], reverse=True)
        best_result = all_results[0]
        result_images = best_result[0]
        
        # Display processing steps for the best iris
        plt.figure(figsize=(15, 8))
        titles = ['Original Eye', 'Enhanced', 'Threshold', 
                 'Iris Mask', 'Iris Region', 'Result']
        
        for i, image in enumerate(result_images):
            plt.subplot(2, 3, i+1)
            if len(image.shape) == 3:
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(image, cmap='gray')
            plt.title(titles[i])
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('iris_processing_steps.jpg')
        plt.show()
        
        # Show full image with all detected eyes and irises
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title("All Detected Eyes and Irises")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('all_detections.jpg')
        plt.show()
        
        # Save the output image with detections
        cv2.imwrite('output_iris_detection.jpg', vis_img)
        
        print(f"Found {len(all_results)} potential iris detections")
        print(f"Best detection score: {best_result[1]:.1f}")
        return result_images
    else:
        print("No suitable iris contour found in any eye region")
        return None

if __name__ == "__main__":
    image_path = "D:/ML/opencv/asd.jpg"
    print(f"Attempting to process image at: {image_path}")
    results = detect_iris_in_eye(image_path)
    
    if results:
        image_names = ["eye", "enhanced", "threshold", "mask", "iris", "result"]
        for i, img in enumerate(results):
            cv2.imwrite(f"{image_names[i]}.jpg", img)
        print("Iris detection complete. Images saved.")
    else:
        print("Failed to detect iris.")
