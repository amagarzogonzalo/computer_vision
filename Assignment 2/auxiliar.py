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

def preprocess_image(image_aux, optimize_image, kernel_params=[(3,3),0.5], canny_thresholds=(375, 375)):
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
    """
    Computes the average background model from a given video.

    :param video_path: Path to the video file.
    :return: The average frame computed from all frames in the video.
    """
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    frames = []
    ret = True

    # Read frames from the video until no more frames are returned
    while ret:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)  # Append each valid frame to the list

    # Compute the average of all frames if the list is not empty
    if frames:
        avg_frame = np.mean(np.array(frames, dtype=np.float32), axis=0).astype(dtype=np.uint8)
    else:
        print("No frames to process.")

    cap.release()  # Release the video capture object
    return avg_frame if frames else None


def subtract_background(frame, background_model):
    """
    Subtracts the background model from a given frame and identifies the foreground.

    :param frame: The current frame from the video.
    :param background_model: The computed background model.
    :return: A mask representing the foreground, contour points, and the largest contour.
    """
    # Convert both frame and background model to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_background = cv2.cvtColor(background_model, cv2.COLOR_BGR2HSV)

    # Calculate the absolute difference between the frame and the background model
    diff = cv2.absdiff(hsv_frame, hsv_background)

    # Threshold the difference in HSV channels to identify significant differences
    _, hue_thresh = cv2.threshold(diff[:, :, 0], 100, 255, cv2.THRESH_BINARY)
    _, sat_thresh = cv2.threshold(diff[:, :, 1], 30, 255, cv2.THRESH_BINARY)
    _, val_thresh = cv2.threshold(diff[:, :, 2], 25, 255, cv2.THRESH_BINARY)

    # Combine thresholds to get a binary mask of the foreground
    combined_mask = cv2.bitwise_or(hue_thresh, cv2.bitwise_or(sat_thresh, val_thresh))

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.erode(combined_mask, kernel, iterations=1)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

    # Find contours and select the largest one as the main foreground object
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        foreground_mask = np.zeros_like(frame)
        cv2.drawContours(foreground_mask, [max_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    else:
        foreground_mask = np.zeros_like(frame)
        max_contour = []

    # Extract contour points from the largest contour
    contour_points = [point for point in max_contour]

    return foreground_mask, contour_points, max_contour


def mog2_method(background_path='background.avi', foreground_path='video.avi', history=100, varThreshold=16, detectShadows=True):
    """
    Applies MOG2 background subtraction method to separate foreground from the background.

    :param background_path: Path to the video file used to model the background.
    :param foreground_path: Path to the video file with the foreground to be extracted.
    :param history: The number of last frames that affect the background model.
    :param varThreshold: Threshold on the squared Mahalanobis distance to decide whether it is well described by
                         the background model. This parameter does not affect the background update.
    :param detectShadows: If true, the algorithm will detect and mark shadows in the output.
    :return: Final mask of the foreground.
    """
    # Initialize video capture for background and foreground videos
    cap = cv2.VideoCapture(background_path)
    cap2 = cv2.VideoCapture(foreground_path)

    # Initialize the background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=detectShadows)

    # Apply the background subtractor to the background video to update the model
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fgbg.apply(frame, learningRate=0.01)

    # Apply the updated model to the foreground video
    while True:
        ret, frame = cap2.read()
        if not ret:
            break
        bg_model = fgbg.apply(frame, learningRate=0)

        # Use multi-Otsu threshold to segment the foreground from the background
        threshold = threshold_multiotsu(bg_model)
        bg_model2 = cv2.threshold(bg_model, threshold[1], 255, cv2.THRESH_BINARY)[1]

        # Find contours and select the largest one
        contours, _ = cv2.findContours(bg_model2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        # Draw the largest contour on a blank mask and apply it to the frame
        final_mask = np.zeros_like(frame)
        cv2.drawContours(final_mask, [max_contour], -1, (255, 255, 255), cv2.FILLED)
        frame = cv2.bitwise_and(frame, final_mask)
        cv2.imshow('Foreground', frame)

    cap.release()
    cap2.release()
    cv2.destroyAllWindows()

    return final_mask