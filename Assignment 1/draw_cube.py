import cv2
import numpy as np

def draw_cube(img, corners, imgpts):
    cube_color = (255, 255, 255)
    imgpts = np.int32(imgpts).reshape(-1, 2)
    img = cv2.drawContours(img, [imgpts[:4]], -1, cube_color, 3)# Draw the base of the cube

    # Draw the sides of the cube
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), cube_color, 3)

    img = cv2.drawContours(img, [imgpts[4:8]], -1, cube_color, 3)# Draw the top of the cube

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
'''

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

'''
