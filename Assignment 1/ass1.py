import cv2
import numpy as np

def load_and_resize_image(image_path, scale_x=0.5, scale_y=0.5):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (0, 0), fx=scale_x, fy=scale_y)
    return resized_image

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        corner_points.append((x, y))
        cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
        cv2.imshow('Image', img)
        aux = x, y
        param.append(aux)

def see_window(window_name, image):
    # See the image completely (resize the window)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
    cv2.resizeWindow(window_name, 500, 500) 
    cv2.imshow(window_name, image)

def find_and_draw_chessboard_corners(image, chessboard_size):
    ret, corners = cv2.findChessboardCorners(image, chessboard_size, None)
    if ret:
        print("Chessboard corners found:")
        print(corners)
        cv2.drawChessboardCorners(image, chessboard_size, corners, ret)
        see_window("Detected Chessboard Corners", image)
    else:
        borders = []
        print("Chessboard corners not found. Click on four corners.")
        see_window("Image", image)
        cv2.setMouseCallback('Image', click_event, borders)
        cv2.waitKey(0)
        borders_np = np.array(borders)
        ret2, corners2 = cv2.findChessboardCorners(image, chessboard_size, borders_np)
        if ret2:
            cv2.drawChessboardCorners(image, chessboard_size, corners2, ret2)
            see_window("Manual Chessboard Corners", image)
        else:
            print("no")



if __name__ == "__main__":
    corner_points = []
    chessboard_size = (6, 9)

    img = load_and_resize_image('images_aux\IMG19.jpg')
    find_and_draw_chessboard_corners(img, chessboard_size) 
    cv2.waitKey(0)  
     

    if not corner_points:
        print("Proceeding with manual corner selection...")
        #interpolate_and_display_points(img)

    cv2.destroyAllWindows()
