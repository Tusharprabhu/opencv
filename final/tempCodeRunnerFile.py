    # Create a figure for subplots
    plt.figure(figsize=(15, 5))

    # Original eye region in grayscale
    plt.subplot(1, 3, 1)
    plt.title("Grayscale Eye Region")
    plt.imshow(eye_roi, cmap='gray')
    plt.axis('off')

    # Original eye region in color
    plt.subplot(1, 3, 2)
    plt.title("Color Eye Region")
    plt.imshow(cv2.cvtColor(eye_color_roi, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Preprocess the eye ROI
    eye_blurred = cv2.GaussianBlur(eye_roi, (7, 7), 0)
    plt.subplot(1, 3, 3)
    plt.title("Blurred Eye Region")
    plt.imshow(eye_blurred, cmap='gray')
    plt.axis('off')

    # Show the combined subplots
    plt.show()