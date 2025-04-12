import cv2
import numpy as np
import sys

def detect_iris_and_pupil(image_path='D:\ML\opencv\S2001R09.jpg'):
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
        roi_color = img[ey:ey+eh, ex:ex+ew].copy()
        roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        
        # Create diagnostic image for viewing intermediate steps
        diagnostic = np.hstack([
            cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR),
            roi_color,
            cv2.cvtColor(roi_blur, cv2.COLOR_GRAY2BGR)
        ])
        
        # IRIS DETECTION
        # Try multiple HoughCircles parameter sets to find the iris
        iris_detected = False
        iris_circle = None
        
        # Parameter sets optimized for iris detection
        iris_parameter_sets = [
            # dp, minDist, param1, param2, minRadius, maxRadius
            [1.5, ew//4, 100, 25, int(ew*0.1), int(ew*0.45)],
            [2.0, ew//4, 100, 20, int(ew*0.1), int(ew*0.45)],
            [2.5, ew//4, 100, 15, int(ew*0.1), int(ew*0.45)]
        ]
        
        for params in iris_parameter_sets:
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
                        iris_circle = circle
                
                if iris_circle is not None:
                    iris_detected = True
                    break
        
        # Draw the detected iris circle
        if iris_circle is not None:
            iris_center = (ex + iris_circle[0], ey + iris_circle[1])
            iris_radius = iris_circle[2]
            
            # Draw iris circle on output image
            cv2.circle(output, iris_center, iris_radius, (0, 255, 0), 2)
            
            # Create a visualization of the iris circle on the eye ROI
            roi_with_iris = roi_color.copy()
            cv2.circle(roi_with_iris, (iris_circle[0], iris_circle[1]), iris_radius, (0, 255, 0), 2)
            
            # PUPIL DETECTION
            # Create a mask for the iris region to search for pupil only within iris
            iris_mask = np.zeros_like(roi_gray)
            cv2.circle(iris_mask, (iris_circle[0], iris_circle[1]), iris_radius, 255, -1)
            
            # Extract iris region and enhance for pupil detection
            iris_roi = cv2.bitwise_and(roi_gray, roi_gray, mask=iris_mask)
            
            # Apply additional processing for pupil detection
            # Pupils are darker than iris, so we can use a lower threshold
            _, pupil_thresh = cv2.threshold(iris_roi, 30, 255, cv2.THRESH_BINARY_INV)
            
            # Clean up the threshold
            kernel = np.ones((3, 3), np.uint8)
            pupil_thresh = cv2.morphologyEx(pupil_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            pupil_thresh = cv2.morphologyEx(pupil_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Add thresholded image to diagnostic
            diagnostic = np.hstack([
                diagnostic, 
                cv2.cvtColor(pupil_thresh, cv2.COLOR_GRAY2BGR)
            ])
            
            # Method 1: Use HoughCircles to find pupil (often works better for clear images)
            pupil_detected = False
            pupil_circle = None
            
            # Parameter sets optimized for pupil detection (smaller circles within iris)
            pupil_parameter_sets = [
                # dp, minDist, param1, param2, minRadius, maxRadius
                [1.5, iris_radius//2, 50, 10, int(iris_radius*0.1), int(iris_radius*0.5)],
                [2.0, iris_radius//2, 40, 10, int(iris_radius*0.1), int(iris_radius*0.5)],
                [2.5, iris_radius//2, 30, 8, int(iris_radius*0.1), int(iris_radius*0.5)]
            ]
            
            for params in pupil_parameter_sets:
                # Only look for pupil within the iris mask
                pupil_circles = cv2.HoughCircles(
                    cv2.bitwise_and(roi_blur, roi_blur, mask=iris_mask),
                    cv2.HOUGH_GRADIENT,
                    dp=params[0],
                    minDist=params[1],
                    param1=params[2],
                    param2=params[3],
                    minRadius=params[4],
                    maxRadius=params[5]
                )
                
                if pupil_circles is not None:
                    # Convert coordinates and radius to integers
                    pupil_circles = np.uint16(np.around(pupil_circles))
                    
                    # Find the most centered circle near the iris center (most likely to be the pupil)
                    iris_center_local = (iris_circle[0], iris_circle[1])
                    min_dist = float('inf')
                    for circle in pupil_circles[0, :]:
                        dist = np.sqrt((iris_center_local[0] - circle[0])**2 + (iris_center_local[1] - circle[1])**2)
                        # Pupil should be near iris center and smaller than iris
                        if dist < min_dist and dist < iris_radius * 0.5 and circle[2] < iris_radius * 0.7:
                            min_dist = dist
                            pupil_circle = circle
                    
                    if pupil_circle is not None:
                        pupil_detected = True
                        break
            
            # Method 2: Use contour detection as backup if HoughCircles fails
            if not pupil_detected:
                contours, _ = cv2.findContours(pupil_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid_contours = []
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 50:  # Minimum area to filter noise
                        # Calculate circularity
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            
                            # Check if the contour is circular
                            if circularity > 0.7:
                                # Get the center and radius
                                (x, y), radius = cv2.minEnclosingCircle(contour)
                                center = (int(x), int(y))
                                radius = int(radius)
                                
                                # Check if within reasonable distance from iris center
                                dist = np.sqrt((iris_circle[0] - center[0])**2 + (iris_circle[1] - center[1])**2)
                                if dist < iris_radius * 0.5 and radius < iris_radius * 0.7:
                                    valid_contours.append((center, radius, circularity, dist))
                
                # Find best pupil candidate
                if valid_contours:
                    # Sort by a combination of circularity and distance to center
                    valid_contours.sort(key=lambda x: (x[2] * 0.7) - (x[3] / iris_radius * 0.3), reverse=True)
                    best_center, best_radius, _, _ = valid_contours[0]
                    
                    pupil_circle = np.array([best_center[0], best_center[1], best_radius])
                    pupil_detected = True
            
            # Draw pupil if detected
            if pupil_circle is not None:
                pupil_center = (ex + pupil_circle[0], ey + pupil_circle[1])
                pupil_radius = pupil_circle[2]
                
                # Draw pupil circle on output image
                cv2.circle(output, pupil_center, pupil_radius, (0, 0, 255), 2)
                # Draw pupil center
                cv2.circle(output, pupil_center, 1, (255, 0, 0), 2)
                
                # Draw pupil on ROI visualization
                cv2.circle(roi_with_iris, (pupil_circle[0], pupil_circle[1]), pupil_radius, (0, 0, 255), 2)
                
                print(f"Iris and pupil detected in largest eye at ({ex},{ey}), size: {ew}x{eh}")
                print(f"Iris radius: {iris_radius}, Pupil radius: {pupil_radius}")
            else:
                print(f"Iris detected but pupil not detected in the largest eye")
                
            # Add final visualization to diagnostic
            diagnostic = np.hstack([
                diagnostic, 
                roi_with_iris
            ])
        else:
            print(f"No iris detected in the largest eye")
    else:
        print("No eyes detected in the image")
    
    # Display results
    cv2.imshow("Iris and Pupil Detection Result", output)
    cv2.imwrite("iris_pupil_detected.jpg", output)
    
    # Display diagnostic images if iris was detected
    if largest_eye is not None and iris_circle is not None:
        # Resize diagnostic if it's too large
        if diagnostic.shape[1] > 1200:
            scale = 1200 / diagnostic.shape[1]
            diagnostic = cv2.resize(diagnostic, None, fx=scale, fy=scale)
        
        cv2.imshow("Detection Steps", diagnostic)
        cv2.imwrite("detection_steps.jpg", diagnostic)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        detect_iris_and_pupil(sys.argv[1])
    else:
        detect_iris_and_pupil()