import cv2
import numpy as np

# Initialize a list to store corner points
corner_points = []


# Function to handle mouse click events
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add the clicked point to the corner_points list
        corner_points.append((x, y))
        # Draw a circle at the clicked point
        cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
        cv2.imshow('Image', img)
        if len(corner_points) == 4:
            print("Selected Corners: ", corner_points)


# Function to interpolate points between the corners
def interpolate_points(p1, p2, num):
    return [(int(p1[0] + (p2[0] - p1[0]) * i / num), int(p1[1] + (p2[1] - p1[1]) * i / num)) for i in range(num + 1)]


# Function to display interpolated chessboard corners
def display_interpolated_corners():
    if len(corner_points) != 4:
        print("Need exactly 4 corner points")
        return

    # Sorting the points to ensure they are ordered correctly
    corner_points_sorted = sorted(corner_points, key=lambda k: [k[1], k[0]])
    top_left, top_right = sorted(corner_points_sorted[:2], key=lambda k: k[0])
    bottom_left, bottom_right = sorted(corner_points_sorted[2:], key=lambda k: k[0])

    num_squares_x = 7  # For an 8x8 chessboard, there are 7 internal corners horizontally
    num_squares_y = 7  # For an 8x8 chessboard, there are 7 internal corners vertically

    all_points = []
    for i in range(num_squares_y + 1):
        start_point = interpolate_points(top_left, bottom_left, num_squares_y)[i]
        end_point = interpolate_points(top_right, bottom_right, num_squares_y)[i]
        row_points = interpolate_points(start_point, end_point, num_squares_x)
        all_points.extend(row_points)

    # Draw all interpolated points
    for point in all_points:
        cv2.circle(img, point, 3, (0, 255, 255), -1)

    cv2.imshow('Interpolated Chessboard Corners', img)


if __name__ == "__main__":
    # Load and display the image
    img = cv2.imread('test2.jpeg')
    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', click_event)

    # Wait until any key is pressed
    cv2.waitKey(0)

    # After 4 corners are selected, display interpolated points
    display_interpolated_corners()

    # Wait until any key is pressed to exit
    cv2.waitKey(0)
    cv2.destroyAllWindows()
