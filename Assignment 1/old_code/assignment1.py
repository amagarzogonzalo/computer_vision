import cv2
import os
from os import listdir

def click_event(event, x, y, flags, params): 
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
        # displaying the coordinates         
        # on the Shell 
        print(x, ' ', y) 
        return x, y
       
print("h")
# get the path/directory
folder_dir = "C:/Users/Alex/Desktop/AI/Workspace/Computer Vision/computer_vision/Assignment 1/images"
for image in os.listdir(folder_dir):
    img = cv2.imread(image, 1) 
    # convert the input image to a grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # displaying the image 
    cv2.imshow('image', img) 
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
    if ret == False: 
        # setting mouse handler for the image 
        # and calling the click_event() function 
        cv2.setMouseCallback('image', click_event) 
    
        # wait for a key to be pressed to exit 
        cv2.waitKey(0) 
        # close the window 
        cv2.destroyAllWindows() 
    else:
        a = 2  
        print("ok")      
print("finish")