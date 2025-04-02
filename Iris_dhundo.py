import cv2
import numpy as np
import sys

def detect_iris(image_path='ab.jpg'):
    # Read image from the given path
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}.")
        sys.exit()
    
    # Create a copy of the original image for output
    output = img.copy()
    
    # Convert to grayscale and apply preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    
    # Apply bilateral filtering to reduce noise while preserving edges
    gray_filtered = cv2.bilateralFilter(gray_eq, 9, 75, 75)
    
    # Load the eye cascade classifier
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Detect eyes with relaxed parameters
    eyes = eye_cascade.detectMultiScale(
        gray_filtered, 
        scaleFactor=1.03, 
        minNeighbors=3, 
        minSize=(20, 20)
    )
    
    # Find the largest eye region
    largest_eye = None
    largest_area = 0
    
    for (ex, ey, ew, eh) in eyes:
        area = ew * eh
        if area > largest_area:
            largest_area = area
            largest_eye = (ex, ey, ew, eh)
    
    # Process only the largest eye if found
    if largest_eye is not None:
        ex, ey, ew, eh = largest_eye
        
        # Draw rectangle around the largest eye
        cv2.rectangle(output, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
        
        # Extract eye region and preprocess
        roi_gray = gray_filtered[ey:ey+eh, ex:ex+ew]
        roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        
        # Try multiple HoughCircles parameter sets to find the iris
        iris_detected = False
        best_circle = None
        
        # Parameter sets optimized for iris detection
        parameter_sets = [
            # dp, minDist, param1, param2, minRadius, maxRadius
            [1.5, ew//4, 100, 25, int(ew*0.1), int(ew*0.45)],
            [2.0, ew//4, 100, 20, int(ew*0.1), int(ew*0.45)],
            [2.5, ew//4, 100, 15, int(ew*0.1), int(ew*0.45)]
        ]
        
        for params in parameter_sets:
            # Apply HoughCircles to find circular iris
            circles = cv2.HoughCircles(
                roi_blur,
                cv2.HOUGH_GRADIENT,
                dp=params[0],
                minDist=params[1],
                param1=params[2],
                param2=params[3],
                minRadius=params[4],
                maxRadius=params[5]
            )
            
            if circles is not None:
                # Convert coordinates and radius to integers
                circles = np.uint16(np.around(circles))
                
                # Find the most centered circle (most likely to be the iris)
                center_x, center_y = ew // 2, eh // 2
                min_dist = float('inf')
                for circle in circles[0, :]:
                    dist = np.sqrt((center_x - circle[0])**2 + (center_y - circle[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_circle = circle
                
                if best_circle is not None:
                    iris_detected = True
                    break
        
        # Draw the detected iris circle
        if best_circle is not None:
            center = (ex + best_circle[0], ey + best_circle[1])
            radius = best_circle[2]
            # Draw circle on output image
            cv2.circle(output, center, radius, (0, 255, 0), 2)
            # Draw circle center
            cv2.circle(output, center, 2, (0, 0, 255), 3)
            print(f"Iris detected in largest eye at ({ex},{ey}), size: {ew}x{eh}")
        else:
            print(f"No iris detected in the largest eye")
    else:
        print("No eyes detected in the image")
    
    # Display result
    cv2.imshow("Iris Detection Result", output)
    cv2.imwrite("iris_detected.jpg", output)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        detect_iris(sys.argv[1])
    else:
        detect_iris()