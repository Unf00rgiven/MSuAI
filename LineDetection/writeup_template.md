Line Detection Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Images of steps are in this zip file.
Link for videos:
https://www.youtube.com/watch?v=8ME6Zjf7-fU&ab_channel=OgnjenStojisavljevic 
https://www.youtube.com/watch?v=x63E7yRW_yk&ab_channel=OgnjenStojisavljevic

Proccesing of video is going in main method in main.py file. For every frame from camera in main method we call output = process_frame(frame).


#### 1. Step 1 - Undistorting image
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` method.  I applied this distortion correction to the test image using the `cv2.undistort()` method and obtained this result.

For every frame process_frame calls undistorted_image = Undisorting.undistort(frame, None)
.If CalibrationType Argument is 'Calibrate' calibrate_camera() will be called.
First images for calibration are loaded and for every image we first get grays_scale_image and then get corners for with cv.findChessboardCorners which we use for cv.CalibrateCamera. It returns  camera_matrix, disorting_coeffs, rvecs, tvecs (rotation and translation vectors). 
After that parameters are saved to file so we dont need to calculate them again which is which is slow and unnecessary.
We calculate once parameters and then just read them.
All those methods save_calibration_parameters, load_calibration_parameters, get_calibration_images, calibrate_camera, undistort are in Undisorting.py file.

#### Step 2 - Binary image
For every frame process_frame calls binary_image = EdgeDetection.filter_lanes_rgb(gausian_image).
The input image is converted from the RGB color space to the HSV color space.Two color ranges are defined, one for yellow and one for white, which are commonly used for lane markings.
Binary masks are created for both the yellow and white colors using the defined thresholds in the HSV image.These two binary masks are combined using a bitwise OR operation to get a single mask that captures both yellow and white lanes.
Code for this step is in EdgeDetection.py file.

#### Step 3 - Perspective transformation
For every frame process_frame calls transformed_image = PerspectiveTransform.warper(binary_image, 'birdPerspective'). We call this method on already binary image and transform it for bird perpespective. Transform points are below. Depending on the transform argument, the method computes a transformation matrix m.
If birdPerspective is specified, the method calculates a transformation to change the viewpoint to a bird's-eye view : m = cv2.getPerspectiveTransform(src, dst). 
If backPerspective is specified, the method calculates the inverse transformation to revert back to the original viewpoint : m = cv2.getPerspectiveTransform(dst, src) (dst and src points are just swapped in thus case).
CV using getPerspectiveTransform and warpPerspective for this purpose.
Code for this step is in PerspectiveTransform.py file.

src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

#### Step 4 - Find lines, draw vehicle pat and get perpestive back
For every frame process_frame calls:
out_img, (left_fitx, right_fitx), (left_fit, right_fit), ploty = LineDetection.sliding_window(transformed_image)
left_curverad, right_curverad, center = LineDetection.get_curve(transformed_image, left_fitx, right_fitx)
lane_image = LineDetection.draw_lanes(frame, left_fitx, right_fitx)
and prints:
print(f"Left curvature: {left_curverad}m")
print(f"Right curvature: {right_curverad}m")
print(f"Center Offset: {center}m")

Histogram Lane Detection (get_hist method):
The method focuses on the bottom half of the image since that's where the lanes are more distinct and closer to the vehicle.
It sums up the white pixels vertically across the image, producing a histogram. Peaks in the histogram indicate the presence of lane lines.

Sliding Window Search (sliding_window method):
This approach divides the image into multiple horizontal windows to isolate and track the lanes.
Initial positions of the lanes are derived from the peaks of the histogram.
Each window then "slides" vertically, adjusting its position based on the concentration of white pixels (potential lane markers) within its boundaries.
After capturing the lane pixels within all windows, a second-order polynomial is fitted to represent the lane.
This polynomial fitting allows the code to predict the lane's trajectory even if some parts of the lane are not clearly visible.
It also keeps a running average of recent polynomial coefficients to smoothen the detection in subsequent frames.

Road Curvature Estimation (get_curve method):
Once lane lines are detected, understanding the curvature becomes essential, especially for autonomous driving, to predict the road's trajectory.
This method calculates the curvature in real-world dimensions (meters) by using a defined conversion from pixel space to real-world space.
It also computes the car's position relative to the center of the lane. This offset gives insights into whether the car is aligned correctly within its lane or drifting towards one side.

Lane Visualization (draw_lanes method):
This method draws the lanes and fills the area between them, representing the drivable path.
It then inverts the perspective transformation to overlay this visual representation on the original road image. This is important because the initial lane detection happens in a bird's-eye view transformed space, which needs to be reverted to the original perspective for display.

Code for this step is in LineDetection.py file.

### Discussion
It appears there is a slight error in detecting the dashed lines, especially near the top. The trajectory and lane detection might not be functioning flawlessly.