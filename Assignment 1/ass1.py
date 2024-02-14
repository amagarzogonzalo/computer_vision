import cv2
import numpy as np
from interpolate import interpolate
import os
from os import listdir


def load_and_resize_image(image_path, scale_x=0.9, scale_y=0.9):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (0, 0), fx=scale_x, fy=scale_y)
    return resized_image

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
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

def find_and_draw_chessboard_corners(image, chessboard_size):
    ret, corners = cv2.findChessboardCorners(image, chessboard_size, None)
    if ret:
        print("Chessboard corners found:", corners)
        cv2.drawChessboardCorners(image, chessboard_size, corners, ret)
        see_window("Detected Chessboard Corners", image)
    else:
        corners = []
        print("Chessboard corners not found. Click on four corners.")
        see_window("Image", image)
        cv2.setMouseCallback('Image', click_event, corners)
        cv2.waitKey(0)
        corners_np = np.array(corners)
        img2 = image

        image, corners_interpolated = interpolate(image, corners, chessboard_size)
        #see_window("Interpolate", img2)
        cv2.drawChessboardCorners(image, chessboard_size, corners_interpolated, ret)

        see_window("Image with linear interpolating", image)




if __name__ == "__main__":
    corner_points = []
    chessboard_size = (6, 9)

    
    folder_dir = 'images_aux2'
    current_dir = os.getcwd()
    folder_path = os.path.join(current_dir, folder_dir)

    for image in os.listdir(folder_dir):
        image_path = os.path.join(folder_path, image)
        print(f"Attempting image: {image}")
        img = load_and_resize_image(image_path)
        find_and_draw_chessboard_corners(img, chessboard_size) 
        cv2.waitKey(0)  


    cv2.destroyAllWindows()
