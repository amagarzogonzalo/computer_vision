import cv2
import numpy as np

def interpolate(image, corners, chessboard_size):
    corners_np = np.float32(corners)
    width, height = 600,900
    dst_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")
        
    step_x = width / (chessboard_size[1] - 1)
    step_y = height / (chessboard_size[0] - 1)
    new_corners_transformed = [(x * step_x, y * step_y) for y in range(chessboard_size[0]) for x in range(chessboard_size[1])]
    new_corners_transformed_np = np.array(new_corners_transformed, dtype='float32').reshape(-1, 1, 2)
    
    # Get the inverse transformation matrix
    matrix_inv = cv2.getPerspectiveTransform(dst_points, corners_np)
    
    # Map the new corners back to the original image space
    new_corners_original = cv2.perspectiveTransform(new_corners_transformed_np, matrix_inv)

    # Draw these corners on the original image
    img_with_corners = image.copy()
    

    cv2.drawChessboardCorners(img_with_corners, chessboard_size, new_corners_original, True)
    
    return img_with_corners, new_corners_original


