import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'D:\ML\opencv\S2002R01.jpg'
img = cv2.imread(image_path)
img1 = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
eyes = eye_cascade.detectMultiScale(gray, scaleFactor=3, minNeighbors=6)

if len(eyes) == 0:
    h, w = gray.shape
    eyes = np.array([[0, 0, w, h]])

for i, (x, y, w, h) in enumerate(eyes):
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    eye_roi = gray[y:y + h, x:x + w]
    eye_color_roi = img[y:y + h, x:x + w]
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    eye_equalized = clahe.apply(eye_roi)
    eye_blurred = cv2.GaussianBlur(eye_equalized, (9, 9), 2)
    iris_circles = cv2.HoughCircles(
        eye_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
        param1=100, param2=30, minRadius=60, maxRadius=150
    )
    if iris_circles is not None:
        iris_circles = np.uint16(np.around(iris_circles))
        for (cx, cy, r) in iris_circles[0, :1]:
            cv2.circle(eye_color_roi, (cx, cy), r, (0, 0, 255), 2)
            iris_mask = np.zeros_like(eye_roi)
            cv2.circle(iris_mask, (cx, cy), int(r * 0.8), 255, -1)
            masked_eye = cv2.bitwise_and(eye_roi, eye_roi, mask=iris_mask)
            _, pupil_thresh = cv2.threshold(masked_eye, 50, 255, cv2.THRESH_BINARY_INV)
            kernel = np.ones((3, 3), np.uint8)
            pupil_morph = cv2.morphologyEx(pupil_thresh, cv2.MORPH_OPEN, kernel)
            pupil_blur = cv2.GaussianBlur(pupil_morph, (5, 5), 1)
            pupil_circles = cv2.HoughCircles(
                pupil_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                param1=50, param2=20, minRadius=max(5, int(r * 0.1)), maxRadius=int(r * 0.6)
            )
            if pupil_circles is not None:
                pupil_circles = np.uint16(np.around(pupil_circles))
                for (pcx, pcy, pr) in pupil_circles[0, :1]:
                    dist_to_center = np.sqrt((pcx - cx)**2 + (pcy - cy)**2)
                    if dist_to_center < r * 0.5:
                        cv2.circle(eye_color_roi, (pcx, pcy), pr, (255, 0, 0), 2)
            else:
                pupil_radius = int(r * 0.3)
                cv2.circle(eye_color_roi, (cx, cy), pupil_radius, (255, 0, 0), 2)
    plt.figure(figsize=(12, 6))
    plt.suptitle(f"Iris and Pupil Detection - Eye #{i+1}", fontsize=16)
    plt.subplot(2, 3, 1)
    plt.title("Original Eye (Color)")
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
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
    plt.subplot(2, 3, 5)
    plt.title("Thresholded Eye")
    plt.imshow(pupil_thresh, cmap='gray')
    plt.subplot(2, 3, 6)
    plt.title("Detected Iris & Pupil")
    plt.imshow(cv2.cvtColor(eye_color_roi, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

cv2.imshow("Final Output - Iris & Pupil Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
