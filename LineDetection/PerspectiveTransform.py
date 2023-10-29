import cv2
import numpy as np

def warper(img, transform):
    # Compute and apply perspective transform
    img_size = (img.shape[1], img.shape[0])
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

    # Argument for transforming and transforming back
    if transform == 'birdPerspective':
        m = cv2.getPerspectiveTransform(src, dst)
    elif transform == 'bachPerspective':
        m = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_NEAREST)

    return warped
