import cv2
import numpy as np
from interpolate import interpolate
from calibration import calibration, undistort, compute_error,extract_frames
import os
from os import listdir

columns = 8
rows = 6
tile_size = 115

def click_event(event, x, y, flags, param):
    """
    Handle mouse click events.

    :param event: The type of mouse event.
    :param x: The x-coordinate of the mouse click.
    :param y: The y-coordinate of the mouse click.
    :param flags: Any flags passed with the event.
    :param param: Additional parameters passed to the function.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        #print(x, ' ', y)
        #cv2.circle(param[1], (x, y), 3, (255, 0, 0), -1)
        #see_window('Image selecting points', param[1])
        aux = x, y
        param[0].append(aux)

def see_window(window_name, image):
    """
    Display an image in a resizable window.

    :param window_name: The name of the window.
    :param image: The image to be displayed.
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 650, 650)
    cv2.imshow(window_name, image)

def task1():

    camera_folders = [os.path.join('Assignment 2', 'data') for folder in os.listdir('Assignment 2') if os.path.isdir(os.path.join('Assignment 2', 'data'))]
    for camera_folder in camera_folders:

        intrinsics_video_path = os.path.join(camera_folder, 'intrinsics.avi')

        # Extract the frames from the video
        frames = extract_frames(intrinsics_video_path)

        selected_frames = frames

        corners = []
        see_window("Image", image)
        cv2.setMouseCallback('Image', click_event, (corners, image))
        cv2.waitKey(0)
        corners, image = interpolate(image, corners, chessboard_size)
        if corners is None or image is None:
            return None
        try:
            corners2 = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)
        except cv2.error as e:
            print("Error in cornerSubPix:", e)
            corners2 = corners
        see_window("Result with Interpolation", image)
        return corners2

