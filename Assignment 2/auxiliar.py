import cv2


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