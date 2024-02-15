import cv2
import numpy as np
from interpolate import interpolate, reverse_again
from calibration import calibration, undistort, compute_error
import os
from os import listdir


def load_and_resize_image(image_path, scale_x=0.9, scale_y=0.9):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (0, 0), fx=scale_x, fy=scale_y)
    return resized_image

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        #print(x, ' ', y)
        corner_points.append((x, y))
        cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
        cv2.imshow('Image', img)
        aux = x, y
        param.append(aux)

def see_window(window_name, image):
    # See the image completely (resize the window)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
    cv2.resizeWindow(window_name, 650, 650) 
    cv2.imshow(window_name, image)

def find_and_draw_chessboard_corners(image, chessboard_size, criteria):
    ret, corners = cv2.findChessboardCorners(image, chessboard_size, None)
    if ret:
        print("Chessboard corners found.")
        corners2 =  corners #cv2.cornerSubPix(image, corners, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        see_window("Detected corners automatically", image)

        return corners2
    else:
        corners = []
        print("Chessboard corners not found. Click on four corners.")
        see_window("Image", image)
        cv2.setMouseCallback('Image', click_event, corners)
        cv2.waitKey(0)
        image_interpolated, corners_interpolated = interpolate(image, corners, chessboard_size)
        if image_interpolated is None or corners_interpolated is None:
            return None
        else: 
            corners2 = corners_interpolated # cv2.cornerSubPix(image_interpolated, corners_interpolated, (11,11), (-1,-1), criteria)
            cv2.drawChessboardCorners(image_interpolated, chessboard_size, corners2, True)
            final_image = reverse_again (image,image_interpolated, corners_interpolated, corners)

            see_window("Result with Interpolation", final_image)
            return corners2

def draw_chessboard_corners(img, corners, chessboard_size):
    ret = 4
    cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
    see_window("Detected Chessboard Corners", img)


if __name__ == "__main__":
    corner_points = []
    chessboard_size = (6, 9)
    square_size = 22
    objpoints = []
    imgpoints = []
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    objp[:,:2]=objp[:,:2]*square_size 

    

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)    
    
    folder_dir = 'images_aux'
    current_dir = os.getcwd()
    folder_path = os.path.join(current_dir, folder_dir)

    for image_i in os.listdir(folder_dir):
        image_path = os.path.join(folder_path, image_i)
        print(f"Attempting image: {image_i}")
        img_aux = load_and_resize_image(image_path)
        # img is gray image
        img = cv2.cvtColor(img_aux, cv2.COLOR_BGR2GRAY)

        corners2 = find_and_draw_chessboard_corners(img, chessboard_size, criteria) 
        if corners2 is not None and len(corners2) > 0: 
            imgpoints.append(corners2)
            objpoints.append(objp)
            cv2.waitKey(0)  
            
            
        else:
            print(f"No corners found for image {image_i}")

    print("Compute Error: ")
    ret, mtx, dist, rvecs, tvecs = calibration(objpoints, imgpoints,  img)
    compute_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)

    cv2.destroyAllWindows()
