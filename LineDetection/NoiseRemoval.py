import cv2

def remove_gaussian_noise(image):
    # Use gausian blur for removing noise
    kernel_size = (7, 7)
    return cv2.GaussianBlur(image, kernel_size, 0)
