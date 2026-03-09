import cv2
import numpy as np

# Create background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define blue color range in HSV
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([150, 255, 255])

    lower_red = np.array([])
    upper_red = np.array([])

    # Create blue mask
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Combine background mask and blue mask to get moving blue objects
    moving_blue = cv2.bitwise_and(blue_mask, blue_mask, mask=fgmask)

    # Clean up mask
    kernel = np.ones((5, 5), np.uint8)
    moving_blue = cv2.morphologyEx(moving_blue, cv2.MORPH_OPEN, kernel)
    moving_blue = cv2.dilate(moving_blue, kernel, iterations=1)

    # Find contours of moving blue objects
    contours, _ = cv2.findContours(moving_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:  # only consider reasonable size objects
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "Blue Object", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Display the frames
    cv2.imshow("Foreground Mask", fgmask)
    cv2.imshow("Blue Foreground Mask", moving_blue)
    cv2.imshow("Detected Blue Objects", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
