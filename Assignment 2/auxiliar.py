import cv2
import numpy as np
import os
from skimage.filters import threshold_multiotsu


def see_window(window_name, image):
    """
    Display an image in a resizable window.

    :param window_name: The name of the window.
    :param image: The image to be displayed.
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 650, 650)
    cv2.imshow(window_name, image)

def extract_frames(video_path, interval=15):
    """
    Extracts frames from a video file every 'interval' frames.
    :param videopath: Path of the video.
    :param interval: Interval used to obtain frames.
    return: Frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Reached end of video
        if count % interval == 0:
            frames.append(frame)
        count += 1
    cap.release()
    return frames


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
        #cv2.circle(param[1], (x, y), 3, (255, 0, 0), -1)
        #see_window('Image selecting points', param[1])
        aux = x, y
        param[0].append(aux)

def save_intrinsics(mtx, rvecs, tvecs, dist, folder):
   
    config_path = os.path.join('data', folder, "intrinsics.xml")
    fs = cv2.FileStorage(config_path, cv2.FILE_STORAGE_WRITE)
    fs.write('mtx', mtx)
    fs.write('rvecs', rvecs)
    fs.write('tvecs', tvecs)
    fs.write('dist', dist)
    print("Intrinsics saved succesfully.")
    fs.release()



def get_intrinsics(folder):
    config_path = os.path.join('data',folder, "intrinsics.xml")
    fs = cv2.FileStorage(config_path, cv2.FILE_STORAGE_READ)

    if not fs.isOpened():
        print("Error: Failed to open intrinsics file.")
        return None

    mtx = fs.getNode('mtx').mat()
    rvecs = fs.getNode('rvecs').mat()
    tvecs = fs.getNode('tvecs').mat()
    dist = fs.getNode('dist').mat()
    fs.release()

    return mtx, dist, rvecs, tvecs

def preprocess_image(image_aux, optimize_image, kernel_params, canny_thresholds):
    """
    Preprocess the input image based on the specified parameters.

    :param image_aux: The input image.
    :param optimize_image: Flag indicating whether to optimize the image.
    :param kernel_params: Parameters for the Gaussian blur kernel (size and sigma).
    :param canny_thresholds: Thresholds for Canny edge detection.
    :return: The preprocessed image.
    """
    if optimize_image:
        gray = cv2.cvtColor(image_aux, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(src=gray, ksize=(kernel_params[0][0], kernel_params[0][1]), sigmaX=kernel_params[1])
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        final_image = clahe.apply(blurred)
        #img = cv2.Canny(blurred, canny_thresholds[0], canny_thresholds[1])
        return final_image, image_aux

    else:
        img = cv2.cvtColor(image_aux, cv2.COLOR_BGR2GRAY)
        return img, image_aux

def averaging_background_model(video_path):

    cap = cv2.VideoCapture(video_path)
    frames = []
    ret = True

    # Read frames from the video
    while ret:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    # Compute the average of frames if the list is not empty
    if frames:
        avg_frame = np.mean(np.array(frames, dtype=np.float32), axis=0).astype(dtype=np.uint8)

        #cv2.imwrite('background_model.jpg', avg_frame)
    else:
        print("No frames to process.")

    # Release the video capture object
    cap.release()
    #print(f'avg frame: {avg_frame}')
    return avg_frame


def subtract_background(frame, background_model):

    # Convert frame to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Convert background
    hsv_background = cv2.cvtColor(background_model, cv2.COLOR_BGR2HSV)
    # Calculate the absolute difference
    diff = cv2.absdiff(hsv_frame, hsv_background)
    #print(f'diff: {diff}')

    # Thresholds
    hue_thresh = 100
    sat_thresh = 30
    val_thresh = 25

    # Apply thresholding
    _, hue_thresh = cv2.threshold(diff[:, :, 0], hue_thresh, 255, cv2.THRESH_BINARY)
    _, sat_thresh = cv2.threshold(diff[:, :, 1], sat_thresh, 255, cv2.THRESH_BINARY)
    _, val_thresh = cv2.threshold(diff[:, :, 2], val_thresh, 255, cv2.THRESH_BINARY)

    # Combine the thresholds to determine foreground (use bitwise operations)
    combined_mask = cv2.bitwise_or(hue_thresh, cv2.bitwise_or(sat_thresh, val_thresh))
    #cv2.imshow("combined_mask",combined_mask)

    # Post-processing
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.erode(combined_mask, kernel, iterations=1)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

    # Find contours and fill the largest one
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        foreground_mask = np.zeros_like(frame)
        cv2.drawContours(foreground_mask, [max_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        #cv2.imshow("Foreground Mask", foreground_mask)
    else:
        foreground_mask = np.zeros_like(frame)
    
    """cv2.imshow("combined_mask",foreground_mask)

    cv2.waitKey(0)"""
    #cv2.destroyAllWindows()
    # Iterate over the points in the contour
    contour_points = []
    for point in max_contour:
        contour_points.append(point)

    return foreground_mask, contour_points, max_contour

def mog2_method(background_path='background.avi', foreground_path = 'video.avi',history=100, varThreshold=16, detectShadows=True):

    cap = cv2.VideoCapture(background_path)
    cap2 = cv2.VideoCapture(foreground_path)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=detectShadows)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _ = fgbg.apply(frame, learningRate=0.01)

    count = 0
    while True:
        ret, frame = cap2.read()
        if not ret:
            break
        bg_model = fgbg.apply(frame, learningRate=0)

        threshold = threshold_multiotsu(bg_model)

        bg_model2 = cv2.threshold(bg_model,threshold[1],255,cv2.THRESH_BINARY)[1]

        # Find the contours in the binary mask
        contours, hierarchy = cv2.findContours(bg_model2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Find the contour with the maximum area
        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        # Create a blank image to draw the contour on
        final_mask = np.zeros_like(frame)

        # Draw the contour with the maximum area on the new mask
        cv2.drawContours(final_mask, [max_contour], -1, (255, 255, 255), cv2.FILLED)

        frame = cv2.bitwise_and(frame, final_mask)

        cv2.imshow('Foreground', frame)

        # Release the camera and close all windows
    cap2.release()
    cv2.destroyAllWindows()
    cap.release()

    return final_mask


