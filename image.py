import cv2

# Load the pre-trained Haar Cascade classifier for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Read the image from file
image_path = 'opencv\imge.jpg'
frame = cv2.imread(image_path)

if frame is None:
    print(f"Error: Could not read image {image_path}.")
    exit()

# Convert the image to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect eyes in the image with adjusted parameters
eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))

# Function to detect the eyeball within the eye region
def detect_eyeball(eye_region):
    # Use edge detection to find contours
    edges = cv2.Canny(eye_region, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour which will be the eyeball
    max_contour = max(contours, key=cv2.contourArea, default=None)
    
    if max_contour is not None:
        # Fit a circle to the largest contour
        (x, y), radius = cv2.minEnclosingCircle(max_contour)
        center = (int(x), int(y))
        radius = int(radius)
        return center, radius
    return None, None

# Draw rectangles around the detected eyes and circles around the eyeballs
for (ex, ey, ew, eh) in eyes:
    cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    eye_region = gray[ey:ey+eh, ex:ex+ew]
    center, radius = detect_eyeball(eye_region)
    if center and radius:
        cv2.circle(frame, (ex + center[0], ey + center[1]), radius, (255, 0, 0), 2)

# Create a named window with the ability to resize
cv2.namedWindow('Eye Detection', cv2.WINDOW_NORMAL)

# Display the output
cv2.imshow('Eye Detection', frame)

# Wait for the user to press the 'q' key or close the window
while True:
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Eye Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

# Close all OpenCV windows
cv2.destroyAllWindows()
