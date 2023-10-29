import numpy as np
import cv2

# Threshold
low_threshold = 50
high_threshold = 150

def get_edges(image, low_threshold=50, high_threshold=150):
    """Applies Canny edge detection on the image."""
    edges = cv2.Canny(image, low_threshold, high_threshold, apertureSize=3)
    return edges
def filter_lanes_rgb(image):
    # Convert the image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Color ranges for yellow and white
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    lower_white = np.array([0, 0, 215])
    upper_white = np.array([180, 40, 255])

    # Masks
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Combination of two masks
    combined_mask = cv2.bitwise_or(yellow_mask, white_mask)

    # Gausian blur for soft edges
    blurred_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)

    return blurred_mask