import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the image
image_path = 'D:\ML\opencv\lkj.jpg'  # Change this to your image path
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 2: Load Haar Cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Step 3: Detect eyes in the grayscale image
eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# Step 4: For each detected eye
for (x, y, w, h) in eyes:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Crop the eye region from the grayscale image
    eye_roi = gray[y:y + h, x:x + w]
    eye_color_roi = img[y:y + h, x:x + w]

    # Create a figure for subplots
    plt.figure(figsize=(15, 15))

    # Original eye region in grayscale
    plt.subplot(3, 3, 1)
    plt.title("Grayscale Eye Region")
    plt.imshow(eye_roi, cmap='gray')
    plt.axis('off')

    # Original eye region in color
    plt.subplot(3, 3, 2)
    plt.title("Color Eye Region")
    plt.imshow(cv2.cvtColor(eye_color_roi, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Preprocess the eye ROI
    eye_blurred = eye_roi.copy()
    plt.subplot(3, 3, 3)
    plt.title("Blurred Eye Region")
    plt.imshow(eye_blurred, cmap='gray')
    plt.axis('off')

    # Get threshold black and white for blurred eye region
    _, eye_thresh = cv2.threshold(eye_blurred, 30, 255, cv2.THRESH_BINARY_INV)
    plt.subplot(3, 3, 4)
    plt.title("Thresholded Eye Region")
    plt.imshow(eye_thresh, cmap='gray')
    plt.axis('off')

    # Detect circles (iris) using HoughCircles
    circles = cv2.HoughCircles(
        eye_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
        param1=100, param2=20, minRadius=5, maxRadius=40
    )

    # Show circles on the thresholded image
    eye_thresh_circles = cv2.HoughCircles(
        eye_thresh, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
        param1=100, param2=20, minRadius=5, maxRadius=40
    )
    plt.subplot(3, 3, 5)
    plt.title("Circles on Thresholded Image")
    plt.imshow(eye_thresh, cmap='gray')
    plt.axis('off')

    # If any circles are found, draw the first one
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (cx, cy, r) in circles[0, :1]:  # Take only the first circle
            cv2.circle(eye_color_roi, (cx, cy), r, (255, 0, 0), 2)

    # Show the detected circles on the color eye region
    plt.subplot(3, 3, 6)
    plt.title("Detected Circles")
    plt.imshow(cv2.cvtColor(eye_color_roi, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Step 5: Show the final image
cv2.imshow("Eye and Iris Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
