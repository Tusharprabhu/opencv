import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_iris_in_eye(image_path):
    # Read image and create eye detector
    img = cv2.imread(image_path)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Convert to grayscale for eye detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    
    # Detect eyes with adjusted parameters
    eyes = eye_cascade.detectMultiScale(
        gray_eq,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(30, 30)
    )
    
    # Find largest eye region
    largest_eye = None
    largest_area = 0
    for (x, y, w, h) in eyes:
        area = w * h
        if area > largest_area:
            largest_area = area
            largest_eye = (x, y, w, h)
    
    if largest_eye is None:
        print("No eye detected!")
        return None
    
    # Extract eye region and create bounding box
    x, y, w, h = largest_eye
    eye_roi = img[y:y+h, x:x+w]
    original = eye_roi.copy()
    
    # Process the eye region for iris detection
    gray_roi = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    equalized = clahe.apply(gray_roi)
    
    # Apply thresholding to isolate darker regions (iris)
    _, thresh = cv2.threshold(equalized, 70, 255, cv2.THRESH_BINARY_INV)
    
    # Clean up the threshold image
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest circular contour (likely the iris)
    max_circularity = 0
    iris_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if 0.6 < circularity < 1.0 and area > 100:  # Check if contour is roughly circular
                if area > max_circularity:
                    max_circularity = area
                    iris_contour = contour
    
    result_images = []
    
    if iris_contour is not None:
        # Fit circle to iris contour
        (ix, iy), radius = cv2.minEnclosingCircle(iris_contour)
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
        
        # Prepare result images
        result_images = [
            original,  # Original eye region
            equalized,  # Enhanced image
            thresh,  # Threshold result
            iris_mask,  # Iris mask
            iris_region,  # Extracted iris
            result  # Final result with circle
        ]
        
        # Draw rectangle on original image
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Draw iris circle on original image
        cv2.circle(img, (x + int(ix), y + int(iy)), radius, (0, 255, 0), 2)
        cv2.circle(img, (x + int(ix), y + int(iy)), 2, (0, 0, 255), -1)
        
        # Display processing steps
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
        
        # Show full image with detected eye and iris
        cv2.imshow('Detected Eye and Iris', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return result_images
    
    return None

if __name__ == "__main__":
    image_path = "D:\ML\opencv\im.jpeg"  # Use your image path
    results = detect_iris_in_eye(image_path)
    if results:
        # Save individual processing steps
        image_names = ["eye", "enhanced", "threshold", "mask", "iris", "result"]
        for i, img in enumerate(results):
            cv2.imwrite(f"{image_names[i]}.jpg", img)
        print("Iris detection complete. Images saved.")
    else:
        print("Failed to detect iris.")