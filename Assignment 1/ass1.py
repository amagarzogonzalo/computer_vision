import cv2
def load_and_resize_image(image_path, scale_x=0.5, scale_y=0.5):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (0, 0), fx=scale_x, fy=scale_y)
    return resized_image

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        corner_points.append((x, y))
        cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
        cv2.imshow('Image', img)


def find_and_draw_chessboard_corners(image, chessboard_size):
    ret, corners = cv2.findChessboardCorners(image, chessboard_size, None)
    if ret:
        print("Chessboard corners found:")
        print(corners)
        cv2.drawChessboardCorners(image, chessboard_size, corners, ret)
        cv2.imshow('Detected Chessboard Corners', image)
    else:
        print("Chessboard corners not found. Click on four corners.")
        cv2.imshow('Image', image)
        cv2.setMouseCallback('Image', click_event)


def interpolate_and_display_points(image, num_squares_x=6, num_squares_y=9):
    if len(corner_points) < 4:
        print("Insufficient points selected.")
        return

    corner_points_sorted = sorted(corner_points, key=lambda k: [k[1], k[0]])
    top_left, top_right = sorted(corner_points_sorted[:2], key=lambda k: k[0])
    bottom_left, bottom_right = sorted(corner_points_sorted[2:], key=lambda k: k[0])

    all_points = []
    for i in range(num_squares_y + 1):
        start_point = interpolate_points(top_left, bottom_left, num_squares_y)[i]
        end_point = interpolate_points(top_right, bottom_right, num_squares_y)[i]
        row_points = interpolate_points(start_point, end_point, num_squares_x)
        all_points.extend(row_points)

    for point in all_points:
        cv2.circle(image, point, 3, (0, 255, 255), -1)

    cv2.imshow('Interpolated Chessboard Corners', image)


def interpolate_points(p1, p2, num):
    return [(int(p1[0] + (p2[0] - p1[0]) * i / num), int(p1[1] + (p2[1] - p1[1]) * i / num)) for i in range(num + 1)]


if __name__ == "__main__":
    corner_points = []
    chessboard_size = (6, 9)

    img = load_and_resize_image('test.jpeg')
    find_and_draw_chessboard_corners(img, chessboard_size)
    cv2.waitKey(0)

    if not corner_points:
        print("Proceeding with manual corner selection...")
        interpolate_and_display_points(img)

    cv2.destroyAllWindows()
