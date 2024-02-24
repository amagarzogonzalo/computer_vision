import cv2
import numpy as np
from interpolate import interpolate
from calibration import calibration, undistort, compute_error
from detect_corners import find_and_draw_chessboard_corners
from auxiliar import see_window, extract_frames, preprocess_image,mog2_method, subtract_background, averaging_background_model
import os
from os import listdir

chessboard_size = 6,8
tile_size = 115


def calibration():
    camera_folders = ["cam1","cam2","cam3","cam4"]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for folder in camera_folders:
        intrinsics_video_path = os.path.join('data', folder, 'intrinsics.avi')
        frames = extract_frames(intrinsics_video_path)
        for frame in frames:
            corners = []
            gray, image = preprocess_image(frame,True, [(3,3),0.5], (375,375))
            see_window("Image", image)
            corners2 = find_and_draw_chessboard_corners(gray, image, chessboard_size, criteria)
            if corners2 is None:
                print("Not corners found for this image.")
            cv2.waitKey(0)


def subtraction():
    camera_folders = ["cam1", "cam2", "cam3", "cam4"]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for folder in camera_folders:
        background_path = os.path.join('data',folder,'background.avi')
        video_path = os.path.join('data',folder,'video.avi')
        frames = extract_frames(video_path,interval = 1)
        '''
        processed_frame = mog2_method(background_path, video_path)
        cv2.imshow('Foreground', processed_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        for frame in frames:
            #MOG2 method
            #processed_frame = mog2_method(background_path, video_path)

            #Average Framing Method
            processed_frame = subtract_background(frame,averaging_background_model(background_path))
            cv2.imshow('Foreground', processed_frame)
            if cv2.waitKey(0):
                break
        cv2.destroyAllWindows()
        

subtraction()
#calibration()