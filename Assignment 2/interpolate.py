import cv2
import numpy as np
from auxiliar import see_window
from calibration import calibrate_camera, compute_error

def lerp(v0, v1, i):
    """
    Performs linear interpolation between two values.

    :param v0: The start value.
    :param v1: The end value.
    :param i: The interpolation factor, where 0 <= i <= 1.
    :return: The interpolated value.
    """
    return v0 + i * (v1 - v0)


def getEquidistantPoints(p1, p2, n):
    """
    Calculates n equidistant points between two points.

    :param p1: The starting point (x, y).
    :param p2: The ending point (x, y).
    :param n: The number of segments to divide the line into.
    :return: A list of equidistant points along the line from p1 to p2.
    """
    return [(lerp(p1[0], p2[0], 1. / n * i), lerp(p1[1], p2[1], 1. / n * i)) for i in range(n + 1)]


def draw_corners(image, eqpoints_x_above, eqpoints_x_bellow, rows):
    """
    Draws auxiliary lines and circles representing the corners on the image.

    :param image: The image on which to draw the auxiliary lines and corners.
    :param eqpoints_x_above: List of points representing the upper edge of the chessboard.
    :param eqpoints_x_bellow: List of points representing the lower edge of the chessboard.
    :param rows: Number of rows in the chessboard grid.
    :return: The annotated image with auxiliary lines and corners.
    """
    auxiliar_line_vertical = []

    for i in range(len(eqpoints_x_above)):
        start_point = (int(eqpoints_x_above[i][0]), int(eqpoints_x_above[i][1]))
        end_point = (int(eqpoints_x_bellow[i][0]), int(eqpoints_x_bellow[i][1]))
        auxiliar_line = getEquidistantPoints(start_point, end_point, rows + 1)
        auxiliar_line = auxiliar_line[1:-1]

        auxiliar_line_vertical.append(auxiliar_line)

    auxiliar_line_vertical = auxiliar_line_vertical[1:-1]

    for i in range(len(auxiliar_line_vertical)):
        line = auxiliar_line_vertical[i]
        cv2.line(image, (int(line[0][0]), int(line[0][1])), (int(line[-1][0]), int(line[-1][1])), (255, 0, 0), 1)
        if i < len(auxiliar_line_vertical) - 1:
            next_line = auxiliar_line_vertical[i + 1]
            cv2.line(image, (int(line[0][0]), int(line[0][1])), (int(next_line[-1][0]), int(next_line[-1][1])),
                     (255, 0, 0), 1)

        for point in line:
            cv2.circle(image, (int(point[0]), int(point[1])), 6, (255, 0, 0))

    return image


def draw_corners_after_transforming(image, corners):
        
    for corner in corners:
        x, y = corner.ravel()
        x = int(x)
        y= int(y)
        cv2.circle(image, (x,y), 1, (0, 255, 255), -1)  
    return image

def interpolate(image, corners, chessboard_size):
    """
    Interpolates and draws a chessboard grid on an image based on provided corner points.

    :param image: The image on which to draw the chessboard.
    :param corners: A list of four corner points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
    :param chessboard_size: A tuple indicating the number of internal corners in the chessboard (rows, rows).
    :return: A tuple containing the modified corner points as a NumPy array and the annotated image.
    """
    if len(corners) < 4:
        print("4 corners are needed to do the interpolation. Number of current points: ", len(corners), ".")
        return None, None

    height, width, _ = image.shape
    origincorners = corners.squeeze()
    corners = np.array(corners, dtype="float32")
    rows, cols= chessboard_size  # 9,6
    """#check order corners is good
    radius = 1
    for corner in origincorners:
        cv2.circle(image, (int(corner[0]), int(corner[1])), radius, (0, 255, 255), thickness=2)
    
        radius += 1"""
    #print(corners)

    dst_points = np.array([[0, height - 1], [width - 1, height - 1], [width - 1, 0],[0,0]], dtype="float32")

    matrix = cv2.getPerspectiveTransform(np.array(corners, dtype="float32"), dst_points)

    axuiliar_warped = cv2.warpPerspective(image, matrix, (width, height))
    corners = dst_points
    eqpoints_x_above = getEquidistantPoints(corners[0], corners[1], rows + 1)
    eqpoints_x_bellow = getEquidistantPoints(corners[3], corners[2], rows + 1)
    eqpoints_y_left = getEquidistantPoints(corners[0], corners[3], cols+ 1)
    eqpoints_y_left = eqpoints_y_left[1:-1]
    eqpoints_y_right = getEquidistantPoints(corners[1], corners[2], cols+ 1)
    eqpoints_y_right = eqpoints_y_right[1:-1]
    auxiliar_line_horizontal = []

    for i in range(len(eqpoints_y_left)):
        cv2.circle(axuiliar_warped, (int(eqpoints_y_left[i][0]), int(eqpoints_y_left[i][1])), 10, (0, 255, 0))
        cv2.circle(axuiliar_warped, (int(eqpoints_y_right[i][0]), int(eqpoints_y_right[i][1])), 10, (255, 255, 0))

        start_point = (int(eqpoints_y_left[i][0]), int(eqpoints_y_left[i][1]))
        end_point = (int(eqpoints_y_right[i][0]), int(eqpoints_y_right[i][1]))
        auxiliar_line = getEquidistantPoints(start_point, end_point, rows+ 1)
        auxiliar_line = auxiliar_line[1:-1]
        auxiliar_line_horizontal.append(auxiliar_line)

    corners_np = np.zeros((cols* rows, 1, 2), dtype=np.float32)
    #print("number of rows and cols", rows, "-", cols)
    for i in range(cols):
        line = auxiliar_line_horizontal[i]
        for j in range(rows):
            point = line[j]
            corners_np[i * rows + j, 0, 0] = point[0]
            corners_np[i * rows + j, 0, 1] = point[1]
            #TODO delete this
            #cv2.circle(axuiliar_warped, (int(point[0]), int(point[1])), 2, (0, 255, 0))


    #see_window('Warped image', axuiliar_warped)
    #cv2.waitKey(0)
    corners_original_image = cv2.perspectiveTransform(corners_np, np.linalg.inv(matrix))


    corners_original_image = np.array(corners_original_image, dtype=np.float32).reshape(-1, 1, 2)

    
    image_after_painting =draw_corners_after_transforming(image, corners_original_image)


    # Testing for interpolation
    """square_size = 22
    objpoints = []
    imgpoints = []
    objp = np.zeros((rows*cols,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:cols].T.reshape(-1,2)
    objp[:,:2]=objp[:,:2]*square_size
    objpoints.append(objp)
    imgpoints.append(corners_original_image)
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(objpoints, imgpoints, image)
    see_window("SUPER AUX", image)
    print("Total error auxiliar:::")
    total_error = compute_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)"""
    #see_window("WORKS?", image_after_painting)
    return corners_original_image, image_after_painting