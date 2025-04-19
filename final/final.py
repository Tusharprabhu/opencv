import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the image
image_path = 'D:/ML/opencv/lkj.jpg'  # Replace backslashes with forward slashes or use raw string
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 2: Load Haar Cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Step 3: Detect eyes in the grayscale image
eyes = eye_cascade.detectMultiScale(gray, scaleFactor=3, minNeighbors=6)

# Step 4: For each detected eye
for i, (x, y, w, h) in enumerate(eyes):
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Crop the eye region from the grayscale and color image
    eye_roi = gray[y:y + h, x:x + w]
    eye_color_roi = img[y:y + h, x:x + w]

    # Step 4.1: Preprocess the eye region
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eye_equalized = clahe.apply(eye_roi)
    eye_blurred = cv2.GaussianBlur(eye_equalized, (9, 9), 2)

    # Step 4.2: Detect iris using HoughCircles
    iris_circles = cv2.HoughCircles(
        eye_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
        param1=100, param2=30, minRadius=20, maxRadius=60
    )

    # Step 4.3: Draw the iris circle
    if iris_circles is not None:
        iris_circles = np.uint16(np.around(iris_circles))
        for (cx, cy, r) in iris_circles[0, :1]:  # Only first detected iris
            cv2.circle(eye_color_roi, (cx, cy), r, (0, 0, 255), 2)  # Red circle

    # Step 4.4: Plot intermediate steps
    plt.figure(figsize=(12, 6))
    plt.suptitle(f"Iris Detection - Eye #{i+1}", fontsize=16)

    plt.subplot(2, 3, 1)
    plt.title("Original Eye (Color)")
    plt.imshow(cv2.cvtColor(eye_color_roi, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("Grayscale Eye")
    plt.imshow(eye_roi, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("Equalized Eye")
    plt.imshow(eye_equalized, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title("Blurred Eye")
    plt.imshow(eye_blurred, cmap='gray')
    plt.axis('off')

    # Optional threshold visualization
    _, eye_thresh = cv2.threshold(eye_blurred, 150, 255, cv2.THRESH_BINARY)
    plt.subplot(2, 3, 5)
    plt.title("Thresholded Eye")
    plt.imshow(eye_thresh, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title("Detected Iris")
    plt.imshow(cv2.cvtColor(eye_color_roi, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Step 5: Final annotated image
cv2.imshow("Final Output - Iris Detection", img) 
cv2.waitKey(0)
cv2.destroyAllWindows()
