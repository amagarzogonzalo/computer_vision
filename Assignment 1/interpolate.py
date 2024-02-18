import cv2
import numpy as np
from itertools import permutations, combinations, product


def interpolate(image, corners, chessboard_size):
    """
    Perform perspective transformation on the input image based on the detected corners of a chessboard.

    :param image: The input image.
    :param corners: The detected corners of the chessboard.
    :param chessboard_size: Size of the chessboard (rows, cols).
    :return: The transformed image and the new corners in the transformed image.
    """
    corners_np = np.float32(corners)
    height, width = image.shape  # 2000, 1500
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")

    try:
        matrix_inv = cv2.getPerspectiveTransform(corners_np, dst_points)
        img_with_corners = cv2.warpPerspective(image, matrix_inv, (width, height))
        step_x = width / (chessboard_size[1] - 1)
        step_y = height / (chessboard_size[0] - 1)
        new_corners_transformed = [(x * step_x, y * step_y) for y in range(chessboard_size[0]) for x in
                                   range(chessboard_size[1])]
        new_corners_transformed_np = np.array(new_corners_transformed, dtype='float32').reshape(-1, 1, 2)

        return img_with_corners, new_corners_transformed_np, matrix_inv

    except:
        print("Wrong number of points. It must be exactly 4 points")
        return None, None


def reverse_again(original_image, interpolated_image, corner_interpolated, corner_original, original_matrix):
    """
    Reverse the perspective transformation applied to the interpolated image and overlay it onto the original image.

    :param original_image: The original image.
    :param interpolated_image: The image after perspective transformation.
    :param corner_interpolated: The corners of the interpolated image.
    :param corner_original: The corners of the original image.
    :return: The overlaid image.
    """
    rows, cols = interpolated_image.shape
    src_pts = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
    dst_pts = np.float32(corner_original)

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped_image = cv2.warpPerspective(interpolated_image, matrix, (original_image.shape[1], original_image.shape[0]))

    mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [dst_pts.astype(np.int32)], (255))

    mask_inv = cv2.bitwise_not(mask)

    original_bg = cv2.bitwise_and(original_image, original_image, mask=mask_inv)
    warped_fg = cv2.bitwise_and(warped_image, warped_image, mask=mask)

    result = cv2.add(original_bg, warped_fg)

    print(corner_original, corner_interpolated)
    rows, cols = interpolated_image.shape[:2]
    corner_original = np.array(corner_original)
    corner_interpolated = np.array(corner_interpolated)

    corresponding_points = np.array([[0, 0],
                                     [cols, 0],
                                     [cols, rows],
                                     [0, rows]])

    corner_interpolated = np.array(corner_interpolated).reshape(-1, 2)
    corner_original = np.array(corner_original).reshape(-1, 2)
    matrixaux, _ = cv2.findHomography(corresponding_points, corner_original)

    scaled_points = cv2.perspectiveTransform(np.array([corner_interpolated], dtype=np.float32), matrixaux)[0]

    scaled_points = np.array(scaled_points)
    scaled_points = scaled_points.reshape(-1, 1, 2)

    print(scaled_points.shape, corner_interpolated.shape)
    sorted_indices = np.argsort(scaled_points[:, 0, 1])
    sorted_scaled_points = scaled_points[sorted_indices]

    return result, sorted_scaled_points