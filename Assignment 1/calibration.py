import cv2
import numpy as np 

def calibration(objpoints, imgpoints, gray):
    """
    The function that calculates the calibration.
    :param objpoints: Object points in the real world.
    :param imgpoints: Image points in the image plane.
    :param gray: The grayscale image used for calibration.
    :return: ret: Boolean indicating successful calibration.
             mtx: Camera matrix.
             dist: Distortion coefficients.
             rvecs: Rotation vectors.
             tvecs: Translation vector
    
    """
    f_x = 0.01
    f_y = 0.01
    cx = gray.shape[1] / 2  
    cy = gray.shape[0] / 2  
    
    mtx = np.array([[f_x, 0, cx],
                    [0, f_y, cy],
                    [0, 0, 1]], dtype=np.float64)
    flags = cv2.CALIB_FIX_FOCAL_LENGTH 

    modify_focal_length = False
    if modify_focal_length:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], mtx, flags)
    else: 
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs

def undistort (img, mtx, dist, image_i):
    """
    Perform undistortion on an image.
    
    :param img: The input image to undistort.
    :param mtx: The camera matrix.
    :param dist: The distortion coefficients.
    :param image_i: Name of the image in the folder. 
    :return: The undistorted image.
    """
    h,  w = img.shape[:2]
    # The new camera matrix for undistortion.
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite(image_i, dst)
    print(f"Image undistort in {image_i}.")


def compute_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    """
    Compute the reprojection error for camera calibration.
    
    :param objpoints: The object points in the real world.
    :param imgpoints: The image points in the image plane.
    :param rvecs: The rotation vectors.
    :param tvecs: The translation vectors.
    :param mtx: The camera matrix.
    :param dist: The distortion coefficients
    :return: The total error
    """ 
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    total_error = mean_error/len(objpoints)
    print(f"Total error: {total_error}. Mean Error: {mean_error}. Number of Object Points (Real World): {len(objpoints)} ")
    return total_error