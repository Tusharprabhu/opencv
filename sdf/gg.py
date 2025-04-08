import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_eye_image(image_path):
    # Step 1: Load the original image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Make a copy for displaying results
    original = gray.copy()
    
    # Step 2: Apply thresholding to isolate the pupil (dark region)
    _, binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    
    # Step 3: Apply edge detection
    edges = cv2.Canny(gray, 30, 150)
    
    # Step 4: Combined processing - apply binary mask to original
    combined = cv2.bitwise_and(gray, binary)
    
    # Step 5 & 6: Circle detection with Hough Transform
    # Apply blur to reduce noise first
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=30, maxRadius=90)
    
    # Create two copies for the different circle visualizations
    result1 = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
    result2 = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:1]:  # Take only the first circle (most prominent)
            # Draw circle for first result
            cv2.circle(result1, (i[0], i[1]), i[2], (0, 255, 0), 2)
            
            # Draw circle for second result (slightly different parameters for demonstration)
            cv2.circle(result2, (i[0], i[1]), i[2], (0, 255, 0), 2)
    
    # Add a red dot at pupil center in the original image (as shown in the reference)
    original_with_dot = cv2.cvtColor(original.copy(), cv2.COLOR_GRAY2BGR)
    if circles is not None:
        i = circles[0,0]
        cv2.circle(original_with_dot, (i[0], i[1]), 3, (0, 0, 255), -1)
    
    # Create a figure with 6 subplots for visualization
    plt.figure(figsize=(12, 8))
    
    # Display all images
    plt.subplot(231)
    plt.imshow(cv2.cvtColor(original_with_dot, cv2.COLOR_BGR2RGB))
    plt.title('1. Original with Center Mark')
    plt.axis('off')
    
    plt.subplot(232)
    plt.imshow(binary, cmap='gray')
    plt.title('2. Binary Threshold')
    plt.axis('off')
    
    plt.subplot(233)
    plt.imshow(edges, cmap='gray')
    plt.title('3. Edge Detection')
    plt.axis('off')
    
    plt.subplot(234)
    plt.imshow(combined, cmap='gray')
    plt.title('4. Combined Mask')
    plt.axis('off')
    
    plt.subplot(235)
    plt.imshow(cv2.cvtColor(result1, cv2.COLOR_BGR2RGB))
    plt.title('5. Circle Detection 1')
    plt.axis('off')
    
    plt.subplot(236)
    plt.imshow(cv2.cvtColor(result2, cv2.COLOR_BGR2RGB))
    plt.title('6. Circle Detection 2')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('eye_processing_steps.jpg')
    plt.show()
    
    # Return all processed images in a list
    return [original_with_dot, binary, edges, combined, result1, result2]

# Main execution
if __name__ == "__main__":
    # Replace with the path to your eye image
    image_path = "D:\ML\opencv\im.jpeg"
    processed_images = process_eye_image(image_path)
    
    # If you want to save individual images:
    image_names = ["original_dot", "binary", "edges", "combined", "circle1", "circle2"]
    for i, img in enumerate(processed_images):
        cv2.imwrite(f"{image_names[i]}.jpg", img)
    
    print("Processing complete. Images saved.")