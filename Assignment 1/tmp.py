
def interpolate3(image, corners, chessboard_size):
    corners_np = np.float32(corners)
    width, height = 650, 650
    dst_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")
    
    # Get the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(corners_np, dst_points)
    
    # Warp the perspective to get the top-down view of the chessboard
    img_with_corners = cv2.warpPerspective(image, matrix, (width, height))
    
    # Calculate the inverse of the perspective transform matrix
    matrix_inv = cv2.getPerspectiveTransform(dst_points, corners_np)
    
    # Create an array of points for the corners in the top-down view
    step_x = width / (chessboard_size[1] - 1)
    step_y = height / (chessboard_size[0] - 1)
    new_corners_transformed = [(x * step_x, y * step_y) for y in range(chessboard_size[0]) for x in range(chessboard_size[1])]
    new_corners_transformed_np = np.array(new_corners_transformed, dtype='float32').reshape(-1, 1, 2)
    
    # Apply the inverse matrix to these points to transform them back to the original image's perspective
    original_corners = cv2.perspectiveTransform(new_corners_transformed_np, matrix_inv)
    
    # Draw the corners on the original image
    cv2.drawChessboardCorners(image, chessboard_size, original_corners, True)
    
    return image, original_corners


def interpolate2(image, corners, chessboard_size):
    corners_np = np.float32(corners)
    width, height = 650,650
    dst_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")
        

    step_x = width / (chessboard_size[1] - 1)
    step_y = height / (chessboard_size[0] - 1)
    new_corners_transformed = [(x * step_x, y * step_y) for y in range(chessboard_size[0]) for x in range(chessboard_size[1])]
    new_corners_transformed_np = np.array(new_corners_transformed, dtype='float32').reshape(-1, 1, 2)
    
    # Get the inverse transformation matrix
    matrix_inv = cv2.getPerspectiveTransform(dst_points, corners_np)
    out = cv2.warpPerspective(image,matrix_inv,(image.shape[1], image.shape[0]),flags=cv2.INTER_LINEAR)
    cv2.imshow('Imageeeee', out)

    
    # Map the new corners back to the original image space
    new_corners_original = cv2.perspectiveTransform(new_corners_transformed_np, matrix_inv)

    # Draw these corners on the original image
    img_with_corners = image.copy()
    

    cv2.drawChessboardCorners(img_with_corners, chessboard_size, new_corners_original, True)
    
    return img_with_corners, new_corners_original
