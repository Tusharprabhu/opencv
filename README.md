### Automated Iris and Pupil Detection in Eye Images Using OpenCV

## Overview

This script performs automated iris and pupil detection in eye images using OpenCV and visualization with Matplotlib. It is designed for processing a single image (specified via `image_path`) and includes several computer vision techniques for robust eye, iris, and pupil localization and visualization.

## Features

- **Eye Detection:** Uses Haar Cascade classifiers to detect eyes in the input image.
- **Fallback:** If no eyes are detected, the whole image is treated as the eye region.
- **Iris Detection:** Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) and Gaussian blur for preprocessing, then uses the Hough Circle Transform to locate the iris.
- **Pupil Detection:** Within the detected iris region, it performs thresholding, morphological operations, and another Hough Circle Transform to locate the pupil.
- **Robustness:** If no pupil is detected, a default position (center of the iris) and radius are used.
- **Visualization:** Displays multiple processing steps (original, grayscale, equalized, blurred, thresholded, detected iris/pupil) using Matplotlib subplots for each detected eye.
- **Final Output:** Shows the annotated image with detected eyes, irises, and pupils using OpenCV's GUI window.

## Usage

- The script reads an image from a hardcoded path (`image_path = 'D:\\ML\\opencv\\S2002R01.jpg'`). Update this path if using a different image.
- The script will process each detected eye in the image and visualize the steps and results.
- For each eye, the following plots are presented:
  - Original Eye (Color)
  - Grayscale Eye
  - Equalized Eye
  - Blurred Eye
  - Thresholded Eye
  - Detected Iris & Pupil (with circles drawn)
- At the end, a window titled "Final Output - Iris & Pupil Detection" shows the annotated full image.

## Main Steps

1. **Load Image:** Reads the input image twice: one for processing and one for display.
2. **Grayscale Conversion:** Converts the image to grayscale for detection.
3. **Eye Detection:** Uses Haar Cascade to find eyes.
4. **Region Handling:** If no eyes are found, the entire image is considered as one eye.
5. **For Each Eye:**
    - Draws a rectangle around the detected eye.
    - Applies CLAHE and Gaussian blur to enhance contrast and reduce noise.
    - Uses Hough Circle Transform to detect the iris.
    - Masks the iris region, thresholds, and morphologically processes to isolate the pupil.
    - Uses Hough Circle Transform again to detect the pupil, with fallback if necessary.
    - Visualizes all steps using Matplotlib subplots.
6. **Final Display:** Shows the processed image with rectangles and circles drawn on detected features.

## Dependencies

- `cv2` (OpenCV)
- `numpy`
- `matplotlib`

## Notes

- The script uses hardcoded image paths and parameters; modify them as needed for other images or use cases.
- All visualization and processing is done in-memory; no files are saved.
- Works best with clear frontal images of eyes.
- Handles cases where no eye or pupil is detected gracefully.

---

For more details on usage or to customize the processing, refer to the code comments and function calls within `final/final.py`.
