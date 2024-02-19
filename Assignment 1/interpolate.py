import cv2
import numpy as np
from itertools import permutations, combinations, product


def lerp(v0, v1, i):
    return v0 + i * (v1 - v0)

def getEquidistantPoints(p1, p2, n):
    return [(lerp(p1[0],p2[0],1./n*i), lerp(p1[1],p2[1],1./n*i)) for i in range(n+1)]


def interpolate(image, corners, chessboard_size):
    rows, cols = chessboard_size
    eqpoints_x_above = getEquidistantPoints(corners[0], corners[1], rows+1)
    eqpoints_x_bellow = getEquidistantPoints(corners[3], corners[2], rows+1)
    eqpoints_y_left = getEquidistantPoints(corners[0], corners[3], cols+1)
    eqpoints_y_right = getEquidistantPoints(corners[1], corners[2], cols+1)

    #cv2.line(image, (int(eqpoints_x_above[1][0]), int(eqpoints_x_above[1][1])), (int(eqpoints_x_bellow[1][0]), int(eqpoints_x_bellow[1][1])), (255, 0, 0), 10)
    mesh_grid = np.zeros((rows+1, cols+1, 2), dtype=np.int32)
    for i in range(rows+1):
        for j in range(cols+1):
            x = int(eqpoints_x_above[j][0] + (eqpoints_x_bellow[j][0] - eqpoints_x_above[j][0]) * (1.0 * i / rows))
            y = int(eqpoints_y_left[i][1] + (eqpoints_y_right[i][1] - eqpoints_y_left[i][1]) * (1.0 * j / cols))
            mesh_grid[i, j] = [x, y]

    # Draw circles at each point in the mesh grid
    for i in range(rows+1):
        for j in range(cols+1):
            cv2.circle(image, (mesh_grid[i, j][0], mesh_grid[i, j][1]), 7, (255, 0, 0), -1)

    for i in eqpoints_x_bellow:
        cv2.circle(image, (int(i[0]), int(i[1])), 3, (255, 0, 0))

    
    # Display image
    window_name = "Interpolated Image"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
    cv2.resizeWindow(window_name, 650, 650)     
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """


    coord_x = [corners[0][0], corners[1][0], corners[2][0], corners[3][0]]
    coord_y = [corners[0][1], corners[1][1], corners[2][1], corners[3][1]]

    xv, yv = np.meshgrid(coord_x, coord_y)
    print(corners,"coordx", coord_x, "coordy",coord_y,xv, yv)

    #aux = np.interp()"""

















def interpolate_aux(image, corners, chessboard_size):
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
        return None, None, None


def reverse_again(original_image, interpolated_image, corner_interpolated, corner_original, original_matrix):
    """
    Reverse the perspective transformation applied to the interpolated image and overlay it onto the original image.

    :param original_image: The original image.
    :param interpolated_image: The image after perspective transformation.
    :param corner_interpolated: The corners of the interpolated image.
    :param corner_original: The corners of the original image.
    :return: The overlaid image and the corners.
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

    #print(corner_original, corner_interpolated)
    scaled_corners = cv2.perspectiveTransform(corner_interpolated, np.linalg.inv(original_matrix))

    return result, scaled_corners