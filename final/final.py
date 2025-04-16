import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the input image using OpenCV
image_path = "D:/ML/opencv/ewtrt.jpg"  # Use forward slashes or raw string
img = cv2.imread(image_path)

# Check if the image was loaded successfully
if img is None:
    print(f"Error: Could not load image at '{image_path}'.")
    exit(1)

# Step 2: Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))#CLAHE (Contrast Limited Adaptive Histogram Equalization)
gray_eq = clahe.apply(gray)

# Step 5: Use the Haar Cascade classifier to detect eyes
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
eyes = eye_cascade.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

# If at least one eye is detected, extract the first one
if len(eyes) > 0:
    (ex, ey, ew, eh) = eyes[0]
    eye_region_color = img[ey:ey + eh, ex:ex + ew]
else:
    print("No eyes detected.")
    exit(1)

# Convert images from BGR (OpenCV) to RGB (matplotlib)
eye_region_rgb = cv2.cvtColor(eye_region_color, cv2.COLOR_BGR2RGB)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(eye_region_rgb)
plt.title("Detected Eye Region")
plt.axis('off')

plt.tight_layout()
plt.show()
