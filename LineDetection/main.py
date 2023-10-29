import cv2

import Undisorting
import EdgeDetection
import PerspectiveTransform
import NoiseRemoval
import LineDetection

def process_frame(frame):
    # Step 1 - Undistortion
    undistorted_image = Undisorting.undistort(frame, None)

    # Step 2 - Noise removal
    gausian_image = NoiseRemoval.remove_gaussian_noise(undistorted_image)

    # Step 3 - Binary image
    binary_image = EdgeDetection.filter_lanes_rgb(gausian_image)

    # Step 4 - Perspective transformation
    transformed_image = PerspectiveTransform.warper(binary_image, 'birdPerspective')

    # Step 5 - Find lines, draw vehicle path and get perspective back
    out_img, (left_fitx, right_fitx), (left_fit, right_fit), ploty = LineDetection.sliding_window(transformed_image)
    left_curverad, right_curverad, center = LineDetection.get_curve(transformed_image, left_fitx, right_fitx)
    lane_image = LineDetection.draw_lanes(frame, left_fitx, right_fitx)

    # Printing the curvature values and  center offset
    print(f"Left curvature: {left_curverad}m")
    print(f"Right curvature: {right_curverad}m")
    print(f"Center Offset: {center}m")

    return lane_image

if __name__ == '__main__':
    # Video input
    video = cv2.VideoCapture(r'C:\Users\Ognjen\PycharmProjects\LineDetection\TestVideos\project_video02.mp4')
    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        output = process_frame(frame)
        cv2.imshow('Line Detection', output)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('Q') or key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
