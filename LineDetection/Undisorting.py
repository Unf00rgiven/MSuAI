import cv2
import numpy as np
import os
import pickle

# OBJP
objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Calibration points
obj_points = []  # 3D points
img_points = []  # 2D points

def save_calibration_parameters(camera_matrix, dist_coeffs, rvecs, tvecs, filename='calibration_parameters.p'):
    # Using this method to save calculated calibration parameters
    with open(filename, 'wb') as f:
        pickle.dump({'camera_matrix': camera_matrix,
                     'dist_coeffs': dist_coeffs,
                     'rvecs': rvecs,
                     'tvecs': tvecs}, f)

def load_calibration_parameters(filename='calibration_parameters.p'):
    # Load saved calibration parameters
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
        rvecs = data['rvecs']
        tvecs = data['tvecs']

    return camera_matrix, dist_coeffs, rvecs, tvecs

def get_calibration_images():
    # Read all images from CameraCalibration folder
    folder_path = r'C:\Users\Ognjen\PycharmProjects\LineDetection\CameraCalibration'
    image_files = [f for f in os.listdir(folder_path)]
    images = []
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)
    return images
def calibrate_camera():
    # Dont need to calibrate camera for every frame, calculate once and then save and use them
    calibration_images = get_calibration_images()
    for image in calibration_images:
        gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pattern_size = (9, 6)
        ret, corners = cv2.findChessboardCorners(gray_scale_image, pattern_size, None)
        if ret:
            obj_points.append(objp)
            img_points.append(corners)
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray_scale_image.shape[::-1], None, None)
    return camera_matrix, dist_coeffs, rvecs, tvecs

def undistort(test_image, calibrationType):
    # If we need to calibrate camera argument will be Calibrate else it will load saved parameters
    if calibrationType == 'Calibrate':
        camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera()
        save_calibration_parameters(camera_matrix, dist_coeffs, rvecs, tvecs)
    else:
        camera_matrix, dist_coeffs, rvecs, tvecs = load_calibration_parameters()

    undistorted_image = cv2.undistort(test_image, camera_matrix, dist_coeffs, None, camera_matrix)

    return undistorted_image