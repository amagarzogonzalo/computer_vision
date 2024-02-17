import cv2
import numpy as np
from interpolate import interpolate, reverse_again
from calibration import calibration, undistort, compute_error
from draw_cube import draw, draw_cube
import os
from os import listdir

def video_mode(mtx, dist, objp, axis):
    i=1
    cam=cv2.VideoCapture(0)
    out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (int(cam.get(3)),int(cam.get(4))))
    while True:
        hasframe,frame=cam.read()
        if hasframe==False:
            break
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        ret,corners = cv2.findChessboardCorners(gray,(9,6),None)
        if ret == True:
            _,rvec,tvec,_=cv2.solvePnPRansac(objp,corners,mtx,dist)
            imgpts,_=cv2.projectPoints(axis,rvec,tvec,mtx,dist)
            frame = draw_cube(frame,corners,imgpts)
        cv2.imshow('images',frame)
        out.write(frame)
        if cv2.waitKey(1)==5:
            break
    cv2.destroyAllWindows()
    cam.release()