import cv2
import numpy as np

# The purpose of the function below is to transform the perspective of a given image which in our case is an image of a
# chessboard, so that a specified corners is remapped to a square of a given size (chessboard_size).
# This function is useful for normalizing the view of a chessboard
def interpolate(image, corners, chessboard_size):
    # Convert corners to floating point for cv2 functions
    corners_np = np.float32(corners)
    # Target image size
    width, height = 600, 600
    # Destination points for perspective transform
    dst_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")

    try:
        # Calculate the perspective transform matrix
        matrix_inv = cv2.getPerspectiveTransform(corners_np, dst_points)
        # Apply perspective warp to the image
        img_with_corners = cv2.warpPerspective(image, matrix_inv, (width, height))
        # Calculate steps for interpolating corners
        step_x = width / (chessboard_size[1] - 1)
        step_y = height / (chessboard_size[0] - 1)
        # Generate new corners after transformation
        new_corners_transformed = [(x * step_x, y * step_y) for y in range(chessboard_size[0]) for x in
                                   range(chessboard_size[1])]
        new_corners_transformed_np = np.array(new_corners_transformed, dtype='float32').reshape(-1, 1, 2)

        return img_with_corners, new_corners_transformed_np

    except:
        print("Wrong number of points. It must be exactly 4 points")
        return None, None


# The purpose of the function below is to take an image that has been interpolated to a new perspective
# and warp it back to its original perspective.
# This is useful for overlaying the processed image back onto the original image.
def reverse_again(original_image, interpolated_image, corner_interpolated, corner_original):
    # Get the size of the interpolated image
    rows, cols, channels = interpolated_image.shape
    # Source points in the interpolated image
    src_pts = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
    # Destination points in the original image
    dst_pts = np.float32(corner_original)

    # Calculate the transformation matrix
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Warp the interpolated image back to its original perspective
    warped_image = cv2.warpPerspective(interpolated_image, matrix, (original_image.shape[1], original_image.shape[0]))

    # Create a mask for the region to overlay on the original image
    mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [dst_pts.astype(np.int32)], (255))

    # Invert mask to subtract the foreground
    mask_inv = cv2.bitwise_not(mask)

    # Extract background from the original image where we want to place the warped image
    original_bg = cv2.bitwise_and(original_image, original_image, mask=mask_inv)
    # Extract the foreground of the warped image
    warped_fg = cv2.bitwise_and(warped_image, warped_image, mask=mask)

    # Combine background and foreground
    result = cv2.add(original_bg, warped_fg)

    return result
