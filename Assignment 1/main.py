import cv2
import numpy as np
from interpolate import interpolate, reverse_again
from calibration import calibration, undistort, compute_error
from draw_cube import draw, draw_cube
import os
from os import listdir
from webcam import webcam_mode
from video import video_mode


def load_and_resize_image(image_path, scale_x=1, scale_y=1):
    """
    Load an image from the given path and resize it.

    :param image_path: The path to the image file.
    :param scale_x: Scaling factor along the x-axis.
    :param scale_y: Scaling factor along the y-axis.
    :return: The resized image.
    """
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (0, 0), fx=scale_x, fy=scale_y)
    return resized_image

def preprocess_image(image_aux, optimize_image, kernel_params, canny_thresholds):
    """
    Preprocess the input image based on the specified parameters.

    :param image_aux: The input image.
    :param optimize_image: Flag indicating whether to optimize the image.
    :param kernel_params: Parameters for the Gaussian blur kernel (size and sigma).
    :param canny_thresholds: Thresholds for Canny edge detection.
    :return: The preprocessed image.
    """
    #print(canny_thresholds)
    if optimize_image:
        gray = cv2.cvtColor(image_aux, cv2.COLOR_BGR2GRAY)


        blurred = cv2.GaussianBlur(src=gray, ksize=(kernel_params[0][0], kernel_params[0][1]), sigmaX=kernel_params[1])
        #img = cv2.Canny(blurred, canny_thresholds[0], canny_thresholds[1])
        return blurred
        see_window("blurred",blurred)
        see_window("canny edge", img)

    else:
        #return image_aux
        img = cv2.cvtColor(image_aux, cv2.COLOR_BGR2GRAY)

    return img

def click_event(event, x, y, flags, param):
    """
    Handle mouse click events.

    :param event: The type of mouse event.
    :param x: The x-coordinate of the mouse click.
    :param y: The y-coordinate of the mouse click.
    :param flags: Any flags passed with the event.
    :param param: Additional parameters passed to the function.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        #print(x, ' ', y)
        #cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
        #cv2.imshow('Image', img)
        aux = x, y
        param.append(aux)

def see_window(window_name, image):
    """
    Display an image in a resizable window.

    :param window_name: The name of the window.
    :param image: The image to be displayed.
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
    cv2.resizeWindow(window_name, 650, 650) 
    cv2.imshow(window_name, image)

def find_and_draw_chessboard_corners(image, chessboard_size, criteria):
    """
    Find and draw chessboard corners on the image.

    :param image: The input image.
    :param chessboard_size: Size of the chessboard (rows, cols).
    :param criteria: Criteria for corner refinement.
    :return: The refined corners.
    """
    ret, corners = cv2.findChessboardCorners(image, chessboard_size, None)
    if ret:
        print("Chessboard corners found.")
        corners2 = cv2.cornerSubPix(image, corners, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(image, chessboard_size, corners2, ret)
        see_window("Detected corners automatically", image)

        return corners2
    else:
        
        corners = []
        print("Chessboard corners not found. Click on four corners.")
        see_window("Image", image)
        cv2.setMouseCallback('Image', click_event, corners)
        cv2.waitKey(0)
        image_interpolated, corners_interpolated = interpolate(image, corners, chessboard_size)
        if image_interpolated is None or corners_interpolated is None:
            return None
        else: 
            final_image = reverse_again (image,image_interpolated, corners_interpolated, corners)

            corners2 = cv2.cornerSubPix(final_image, corners_interpolated, (11,11), (-1,-1), criteria)
            cv2.drawChessboardCorners(image_interpolated, chessboard_size, corners2, True)
            final_image = reverse_again (image,image_interpolated, corners2, corners)
            see_window("Result with Interpolation", final_image)
            return corners2


def online_phase(img,optimize_image, kernel_params, canny_params, chessboard_size, criteria, mtx, dist, rvecs, tvecs, objp):
    """
    Perform the online phase for a test image.

    :param img: The test image.
    :param optimize_image: Flag indicating whether to optimize the image.
    :param kernel_params: Parameters for the Gaussian blur kernel (size and sigma).
    :param canny_params: Thresholds for Canny edge detection.
    :param chessboard_size: Size of the chessboard (rows, cols).
    :param criteria: Criteria for corner refinement.
    :param mtx: The camera matrix.
    :param dist: The distortion coefficients.
    :param rvecs: Rotation vectors.
    :param tvecs: Translation vectors.
    :param objp: The object points representing the 3D corners of the chessboard.
    """
    print("Online phase for Test Image:")
    #test_image_path = os.path.join(os.getcwd(), 'test', 'IMG20.jpg')
    #img_aux = load_and_resize_image(test_image_path)
    #img = preprocess_image (img_aux, False, kernel_params, canny_params)
    #img_aux = load_and_resize_image(img)
    #img_aux =
    img = preprocess_image(img, False, kernel_params, canny_params)
    corners2 = find_and_draw_chessboard_corners(img, chessboard_size, criteria)
    square_size = 22  # Size of a chessboard square in mm
    #for cube line axis
    axis = np.float32([
        [0, 0, 0], [0, square_size * 3, 0], [square_size * 3, square_size * 3, 0], [square_size * 3, 0, 0],  # Base
        [0, 0, -square_size * 3], [0, square_size * 3, -square_size * 3],
        [square_size * 3, square_size * 3, -square_size * 3], [square_size * 3, 0, -square_size * 3]  # Top
    ])
    #for axis lines
    axis2 = np.float32([[0,0,0],[square_size*3,0,0],[0,square_size*3,0],[0,0,square_size*-3]])

    if corners2 is not None and len(corners2) > 0:
        _, rvec, tvec, _ = cv2.solvePnPRansac(objp, corners2, mtx, dist)
        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
        imgpts2, _ = cv2.projectPoints(axis2,rvec,tvec,mtx,dist)
        color_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        color_image = draw_cube(color_image, corners2, imgpts)
        color_image = draw(color_image, corners2, imgpts2)
        see_window("Image with cube and axis lines", color_image)
        #print(f'imgpts: {imgpts}')
        #print(f'imgpts: {imgpts2}')
        print("Online phase done.")
        cv2.waitKey(0)

    else:
        print("No corners found in the test image.")

def run(select_run, optimize_image, kernel_params, canny_params, webcam, video):
    """
    Run the calibration process.

    :param select_run: The selected run number.
    :param optimize_image: Flag indicating whether to optimize the image.
    :param kernel_params: Parameters for the Gaussian blur kernel (size and sigma).
    :param canny_params: Thresholds for Canny edge detection.
    :param webcam: Flag indicating whether to use webcam mode.
    :param video: Flag indicating whether to record a video.
    """
    corner_points = []
    chessboard_size = (6, 9)
    square_size = 22
    objpoints = []
    imgpoints = []
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)
    objp[:,:2]=objp[:,:2]*square_size

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    if webcam == 1:
        webcam_mode()
        folder_dir = 'webcam'
    elif select_run == 1:
        folder_dir = 'run_1'
        print("Run 1:")
    elif select_run == 2:
        folder_dir = 'run_2'
        print("Run 2:")
    elif select_run == 3:
        folder_dir = 'run_3'
        print("Run 3:")
    elif select_run == 0:
        folder_dir = 'images_aux2'
        print("Auxiliar Run")
    else:
        folder_dir = 'run_1'
        print("Run 1:")
    current_dir = os.getcwd()
    folder_path = os.path.join(current_dir, folder_dir)
    fail = False

    for image_i in os.listdir(folder_dir):
        image_path = os.path.join(folder_path, image_i)
        print(f"Attempting image: {image_i}.")
        img_aux = load_and_resize_image(image_path)
        # img is gray image
        img = preprocess_image (img_aux, optimize_image, kernel_params, canny_params)
       
        corners2 = find_and_draw_chessboard_corners(img, chessboard_size, criteria) 
        if corners2 is not None and len(corners2) > 0: 
            imgpoints.append(corners2)
            objpoints.append(objp)
            cv2.waitKey(0)
            ret, mtx, dist, rvecs, tvecs = calibration(objpoints, imgpoints, img)
            online_phase(img_aux,optimize_image, kernel_params, canny_params, chessboard_size, criteria, mtx, dist, rvecs,tvecs, objp)
            print(f'Camera Matrix (K): {mtx}')
            print(f'Image resolution: {img_aux.shape[1]}x{img_aux.shape[0]}')

            if video == 1:
                #ret, mtx, dist, rvecs, tvecs = calibration(objpoints, imgpoints, img)
                #square_size = 21  # Size of a chessboard square in mm
                axis = np.float32([
                    [0, 0, 0], [0, square_size * 3, 0], [square_size * 3, square_size * 3, 0], [square_size * 3, 0, 0],# Base
                    [0, 0, -square_size * 3], [0, square_size * 3, -square_size * 3],
                    [square_size * 3, square_size * 3, -square_size * 3], [square_size * 3, 0, -square_size * 3]  # Top
                ])
                video_mode(mtx, dist, objp, axis)



        else:
            print(f"No corners found for image {image_i}.")
            fail = True

    if not fail:
        print("Compute Error: ")
        ret, mtx, dist, rvecs, tvecs = calibration(objpoints, imgpoints,  img)
        if ret:
            total_error = compute_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
            cv2.destroyAllWindows()
            online_phase(img_aux,optimize_image, kernel_params, canny_params, chessboard_size, criteria, mtx, dist, rvecs,tvecs,objp)
            return total_error

        else:
            print("Error during calibration.")
            cv2.destroyAllWindows()
            return 0
    else:
        return 0

def main():
    select_run = 3
    webcam = 0
    video = 1
    optimize_image = False
    kernel_params = [(3,3),0.5]
    canny_params = (375, 375)
    #run(select_run=1)
    #run(select_run=2)
    run(select_run, optimize_image, kernel_params, canny_params, webcam, video)

main()