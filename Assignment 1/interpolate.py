import cv2
import numpy as np

def interpolate(image, corners, chessboard_size):
    corners_np = np.float32(corners)
    width, height = 600, 600  
    dst_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(corners_np, dst_points)
    img_transformed = cv2.warpPerspective(image, matrix, (width, height))
    step_x = width / (chessboard_size[1] - 1)
    step_y = height / (chessboard_size[0] - 1)
    new_corners = [(x * step_x, y * step_y) for y in range(chessboard_size[0]) for x in range(chessboard_size[1])]
    
    return img_transformed, np.array(new_corners, dtype='float32').reshape(-1, 1, 2)
