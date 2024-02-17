import cv2
import numpy as np

def draw_cube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),3)
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

def draw(img, corners, imgpts):
    # Ensure the points are in the correct format (integer tuples)
    origin = tuple(imgpts[0].ravel().astype(int))
    x_axis_end = tuple(imgpts[1].ravel().astype(int))
    y_axis_end = tuple(imgpts[2].ravel().astype(int))
    z_axis_end = tuple(imgpts[3].ravel().astype(int))

    # Draw the lines for the X, Y, Z axes
    img = cv2.line(img, origin, x_axis_end, (0, 0, 255), 5)  # X-axis in red
    img = cv2.line(img, origin, y_axis_end, (0, 255, 0), 5)  # Y-axis in green
    img = cv2.line(img, origin, z_axis_end, (255, 0, 0), 5)  # Z-axis in blue
    return img
