import cv2
import numpy as np

corner_points = []
chessboard_size = (6,9)


test = cv2.imread('test2.jpeg')
#grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(test, (0, 0), fx = 0.5, fy = 0.5)

def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        return x, y

def check_corner(img):
    ret, corners = cv2.findChessboardCorners(img, chessboard_size, None)
    print("a")
    if ret:
        print("b")
        print("Chessboard corners found:")
        print(corners)

        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Detected Chessboard Corners', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("c")
        print("Chessboard corners not found.")

check_corner(img)

