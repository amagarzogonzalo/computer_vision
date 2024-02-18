import cv2
import numpy as np
from interpolate import interpolate, reverse_again
from calibration import calibration, undistort, compute_error
from draw_cube import draw, draw_cube
import os
from os import listdir

def video_mode(mtx, dist, objp, axis):
    """
    This function captures video from a camera, detects a chessboard pattern in each frame, and overlays a 3D cube on the detected chessboard.

    :param mtx: The camera matrix.
    :param dist: The distortion coefficients.
    :param objp: The object points representing the 3D corners of the chessboard.
    :param axis: The axis points representing the 3D cube to overlay on the chessboard.
    """
    cam = cv2.VideoCapture(0)
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (int(cam.get(3)), int(cam.get(4))))

    while True:
        hasframe, frame = cam.read()
        if not hasframe:
            break

        # Flip the frame to correct the mirroring effect
        flipped_frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (6,9), None)

        if ret:
            _, rvec, tvec, _ = cv2.solvePnPRansac(objp, corners, mtx, dist)
            imgpts, _ = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
            # Make sure to draw on the flipped frame
            flipped_frame = draw_cube(flipped_frame, corners, imgpts)

        # Display the flipped frame
        cv2.imshow('images', flipped_frame)
        # Write the flipped frame to the output file
        out.write(flipped_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Use ord('q') to exit
            break

    cam.release()
    out.release()
    cv2.destroyAllWindows()
