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

def preprocess_image(image_aux, optimize_image, kernel_params):
    print(kernel_params)
    if optimize_image:
        #img = cv2.GaussianBlur(img, (3, 3), 0)
        gray = cv2.cvtColor(image_aux, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(src=gray, ksize=(kernel_params[0][0], kernel_params[0][1]), sigmaX=kernel_params[1])
        #img = cv2.Canny(blurred, 70, 135)
        return blurred
        see_window("blurred",blurred)
        see_window("canny edge", img)

    else: 
        img = cv2.cvtColor(image_aux, cv2.COLOR_BGR2GRAY)

    return img



def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        #print(x, ' ', y)
        #cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
        #cv2.imshow('Image', img)
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
        cv2.drawChessboardCorners(image, chessboard_size, corners2, ret)
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


def run(select_run, optimize_image, kernel_params):
    corner_points = []
    chessboard_size = (6, 9)
    square_size = 22
    objpoints = []
    imgpoints = []
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)
    objp[:,:2]=objp[:,:2]*square_size 

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)    
    
    if select_run == 1:
        folder_dir = 'run_1'
    elif select_run == 2:
        folder_dir = 'run_2'
    elif select_run == 3:
        folder_dir = 'run_3'
    elif select_run == 0:
        folder_dir = 'images_aux'
    else:
        folder_dir = 'run_1'
    current_dir = os.getcwd()
    folder_path = os.path.join(current_dir, folder_dir)

    for image_i in os.listdir(folder_dir):
        image_path = os.path.join(folder_path, image_i)
        print(f"Attempting image: {image_i}.")
        img_aux = load_and_resize_image(image_path)
        # img is gray image
        img = preprocess_image (img_aux, optimize_image, kernel_params)
       
        corners2 = find_and_draw_chessboard_corners(img, chessboard_size, criteria) 
        if corners2 is not None and len(corners2) > 0: 
            imgpoints.append(corners2)
            objpoints.append(objp)
            cv2.waitKey(0)  
            if select_run == 3:
                ret, mtx, dist, rvecs, tvecs = calibration(objpoints, imgpoints,  img)
                if ret:
                    compute_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
                else:
                    print("Error during calibration.")
            
            
        else:
            print(f"No corners found for image {image_i}.")

    print("Compute Error: ")
    ret, mtx, dist, rvecs, tvecs = calibration(objpoints, imgpoints,  img)
    if ret:
        total_error = compute_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
        cv2.destroyAllWindows()
        return total_error

    else:
        print("Error during calibration.")
        cv2.destroyAllWindows()
        return 0





def main():
    select_run = 2
    optimize_image = False
    kernel_params = [(3,5),0.5]
    #run(select_run=1)
    #run(select_run=2)
    run(select_run, optimize_image, kernel_params)

#main()