import cv2
import numpy as np

def interpolate(image, corners, chessboard_size):
    corners_np = np.float32(corners)
    width, height = 600,600
    dst_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")
    
    try:
        matrix_inv = cv2.getPerspectiveTransform(corners_np, dst_points)
        img_with_corners = cv2.warpPerspective(image,matrix_inv,(width, height))
        step_x = width / (chessboard_size[1] - 1)
        step_y = height / (chessboard_size[0] - 1)
        new_corners_transformed = [(x * step_x, y * step_y) for y in range(chessboard_size[0]) for x in range(chessboard_size[1])]
        new_corners_transformed_np = np.array(new_corners_transformed, dtype='float32').reshape(-1, 1, 2)

        return img_with_corners, new_corners_transformed_np

    except :
        print("Wrong number of points. It must be exactly 4 points") 
        return None, None
    

def reverse_again(original_image, interpolated_image, corner_interpolated, corner_original):

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
    
    return result