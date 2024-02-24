import cv2
import numpy as np
from interpolate import interpolate
from calibration import calibrate_camera, undistort, compute_error
from detect_corners import find_and_draw_chessboard_corners, detect_corners_automatically, draw_corners, detect_corners_for_extrinsic
from auxiliar import see_window, extract_frames, preprocess_image, mog2_method, subtract_background, averaging_background_model, save intrinsics
import os
from os import listdir

chessboard_size = 6,8
tile_size = 115


def camera_intrinsic():
    frames_per_folder = 1
    camera_folders = ["cam1","cam2","cam3","cam4"]
    interval = 10
    camera_folders = ["cam1"]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    rows, cols = chessboard_size
    square_size = 22
    objpoints = []
    imgpoints = []
    objp = np.zeros((rows*cols,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:cols].T.reshape(-1,2)
    objp[:,:2]=objp[:,:2]*square_size
    first_frame = None
    for folder in camera_folders:
        cont_aux = 0

        print("Checking folder: ", folder)
        intrinsics_video_path = os.path.join('data', folder, 'intrinsics.avi')
        frames = extract_frames(intrinsics_video_path, interval)
        for frame in frames:
            gray, image = preprocess_image(frame,True, [(3,3),0.5], (375,375))
            if first_frame is None:
                first_frame = image
            corners2, image = find_and_draw_chessboard_corners(gray, image, chessboard_size, criteria, interval, do_manual=False, skip_manual=True)
            if corners2 is None or image is None:
                print("Not corners found for this image.")
            else:
                if len(corners2) == 2:
                    imgpoints.append(corners2[0])
                else:
                    imgpoints.append(corners2)
                objpoints.append(objp)
                cont_aux += 1
                if cont_aux > frames_per_folder-1:
                    break
          
    #cv2.waitKey(0)
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(objpoints, imgpoints, first_frame)
    if ret:
        total_error = compute_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
        cv2.destroyAllWindows()
        return total_error, mtx,dist, rvecs,tvecs


def camera_extrinsic(mtx, dist, rvec, tvec):
    camera_folders = ["cam1","cam2","cam3","cam4"]
    interval = 10
    camera_folders = ["cam1"]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    rows, cols = chessboard_size
    square_size = 22
    objpoints = []
    imgpoints = []
    objp = np.zeros((rows*cols,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:cols].T.reshape(-1,2)
    objp[:,:2]=objp[:,:2]*square_size
    first_frame = None
    for folder in camera_folders:
        cont_aux = 0
        print("Checking folder: ", folder)
        intrinsics_video_path = os.path.join('data', folder, 'checkerboard.avi')
        frames = extract_frames(intrinsics_video_path, interval)
        gray, image = preprocess_image(frames[0],True, [(3,3),0.5], (375,375))
        #corners = detect_corners_for_extrinsic(gray,image,chessboard_size)
        #image = draw_corners(image, corners)
        corners2, image = find_and_draw_chessboard_corners(gray, image, chessboard_size, criteria, interval, do_manual=True, skip_manual=False)
        if corners2 is None or image is None:
            print("Not corners found for this image.")
        else:
            see_window("Interpolation done.", image)
            cv2.waitKey(0)
            print("Rotating.")
            ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
            corners3 = corners2.astype(int)

            square_size = 4  # Size of a chessboard square in mm
            #for axis lines
            size_of_axis = square_size*3
            #axis2 = np.float32([[0,0,0],[square_size*3,0,0],[0,square_size*3,0],[0,0,square_size*-3]])

            axis2 = np.float32([[size_of_axis,0,0],[0,size_of_axis,0],[0,0,-size_of_axis],[0,0,0]])
            axis_points, _ = cv2.projectPoints(axis2*15,rvecs,tvecs,mtx,dist)
            axis_points = np.round(axis_points).astype(int)
    
        
            for i in range(3):
                image = cv2.line(image, tuple(corners3[0].ravel()), tuple(axis_points[i].ravel()), ((0,255, 0), (255, 0, 0), (0, 0, 255))[i], 3)
            see_window("Axis", image)
            save_intrinsics(mtx,rvecs,tvecs,dist)
            
            
            
            cv2.waitKey(0)

def subtraction():
    camera_folders = ["cam1", "cam2", "cam3", "cam4"]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for folder in camera_folders:
        background_path = os.path.join('data',folder,'background.avi')
        video_path = os.path.join('data',folder,'video.avi')
        frames = extract_frames(video_path,interval = 1)
        for frame in frames:
            #mog2 method
            #processed_frame = mog2_method(background_path, video_path)

            #avergage fram method
            processed_frame = subtract_background(frame, averaging_background_model(video_path))
            cv2.imshow('Foreground', processed_frame)
            if cv2.waitKey(0):
                break
        cv2.destroyAllWindows()



#subtraction()
total_error, mtx,dist, rvecs,tvecs = camera_intrinsic()
#camera_extrinsic(mtx=None,dist=None, rvec=None, tvec=None)
camera_extrinsic(mtx,dist, rvecs, tvecs)