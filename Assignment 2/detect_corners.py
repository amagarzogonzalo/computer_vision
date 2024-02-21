import cv2
import numpy as np
from interpolate import interpolate
from auxiliar import see_window, click_event

def corners_sub_pix(image, corners, criteria):
    try:
        corners2 = cv2.cornerSubPix(image, corners, (11,11), (-1,-1), criteria)
    except cv2.error as e:
        print("Error in cornerSubPix:", e)
        corners2 = corners   
    return corners2


def detect_corners_automatically(gray, img, number_corners=48, threshold= 0.1, min_ec_distance=10):
    corners = cv2.goodFeaturesToTrack(gray,number_corners,threshold, min_ec_distance)
    corners = np.int0(corners)
    if len(corners) < 20:
        return False, None, None
    else:
        for i in corners:
            x,y = i.ravel()
            cv2.circle(img,(x,y),3,255,-1)
        return True, corners, img  

def find_and_draw_chessboard_corners(gray, image, chessboard_size, criteria):
    """
    Find and draw chessboard corners on the image.


    :param gray: The input image in gray format.
    :param image: The input image.    
    :param chessboard_size: The size of the chessboard.
    :param criteria: Criteria for corner refinement.
    :return: The refined corners.
    """
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        print("Chessboard corners found.")
        corners2 = corners_sub_pix(gray,corners,criteria)            
        cv2.drawChessboardCorners(image, chessboard_size, corners2, ret)
        see_window("Detected corners automatically", image)
        return corners2, image
    else:
        
        done, corners, image = detect_corners_automatically(gray, image)
        corners = []
        if not done:
            print("Chessboard corners not found. Click on four corners.")
            see_window("Image", image)
            cv2.setMouseCallback('Image', click_event,  (corners, image))
            cv2.waitKey(0)
            corners, image = interpolate(image, corners, chessboard_size)
            if corners is None or image is None:
                return None
            corners2 = corners_sub_pix(gray,corners,criteria)            
            see_window("Result with Interpolation", image)
            return corners2, image
        else:
            corners2 = corners_sub_pix(gray,corners,criteria)            
            see_window("Corners Detected Automatically", image)
            return corners2, image
