import cv2
import numpy as np


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


def draw_corners(image, eqpoints_x_above, eqpoints_x_bellow, cols):
    """
    Draws auxiliary lines and circles representing the corners on the image.

    :param image: The image on which to draw the auxiliary lines and corners.
    :param eqpoints_x_above: List of points representing the upper edge of the chessboard.
    :param eqpoints_x_bellow: List of points representing the lower edge of the chessboard.
    :param cols: Number of columns in the chessboard grid.
    :return: The annotated image with auxiliary lines and corners.
    """
    auxiliar_line_vertical = []

    for i in range(len(eqpoints_x_above)):
        start_point = (int(eqpoints_x_above[i][0]), int(eqpoints_x_above[i][1]))
        end_point = (int(eqpoints_x_bellow[i][0]), int(eqpoints_x_bellow[i][1]))
        auxiliar_line = getEquidistantPoints(start_point, end_point, cols + 1)
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


def interpolate(image, corners, chessboard_size):
    """
    Interpolates and draws a chessboard grid on an image based on provided corner points.

    :param image: The image on which to draw the chessboard.
    :param corners: A list of four corner points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
    :param chessboard_size: A tuple indicating the number of internal corners in the chessboard (rows, cols).
    :return: A tuple containing the modified corner points as a NumPy array and the annotated image.
    """
    if len(corners) < 4:
        print("4 corners are needed to do the interpolation.")
        return None, None
    cols, rows = chessboard_size  # 9,6

    eqpoints_x_above = getEquidistantPoints(corners[0], corners[1], cols + 1)
    eqpoints_x_bellow = getEquidistantPoints(corners[3], corners[2], cols + 1)
    eqpoints_y_left = getEquidistantPoints(corners[0], corners[3], rows + 1)
    eqpoints_y_left = eqpoints_y_left[1:-1]
    eqpoints_y_right = getEquidistantPoints(corners[1], corners[2], rows + 1)
    eqpoints_y_right = eqpoints_y_right[1:-1]
    auxiliar_line_horizontal = []

    for i in range(len(eqpoints_y_left)):
        start_point = (int(eqpoints_y_left[i][0]), int(eqpoints_y_left[i][1]))
        end_point = (int(eqpoints_y_right[i][0]), int(eqpoints_y_right[i][1]))
        auxiliar_line = getEquidistantPoints(start_point, end_point, rows + 1)
        auxiliar_line = auxiliar_line[1:-1]
        auxiliar_line_horizontal.append(auxiliar_line)

    corners_np = np.zeros((rows * cols, 1, 2), dtype=np.float32)

    for i in range(rows):
        line = auxiliar_line_horizontal[i]
        for j in range(cols):
            point = line[j]
            corners_np[i * cols + j, 0, 0] = point[0]
            corners_np[i * cols + j, 0, 1] = point[1]

    corners_np = np.array(corners_np, dtype=np.float32).reshape(-1, 1, 2)

    return corners_np, draw_corners(image, eqpoints_x_above, eqpoints_x_bellow, cols)

    # CODE2
    corners_np = np.array(auxiliar_line_vertical, dtype=np.float32).reshape(-1, 1, 2)
    sorted_indices = np.argsort(corners_np[:, :, 1].flatten())

    corners_modified = corners_np[sorted_indices]
    # print("Corners2",corners_modified)

    return corners_modified, image