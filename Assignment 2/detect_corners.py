import cv2
import numpy as np
from interpolate import interpolate, getEquidistantPoints
from auxiliar import see_window, click_event, preprocess_image
from scipy.spatial.distance import cdist
import math
from scipy.spatial import distance


class Outsider_corners:
    calculated = False
    distance_threshold = 0
    top_left = (0,0)
    top_right = (0,0)
    bottom_left = (0,0)
    bottom_right = (0,0)

def detect_corners_for_extrinsic(gray, img, chessboard_size, number_corners=48, threshold= 0.005, min_ec_distance=5):
    corners = cv2.goodFeaturesToTrack(gray,number_corners,threshold, min_ec_distance, useHarrisDetector=True, k=0.005)
    corners = np.int0(corners)
    #see_window("Aux", img)
    return corners

def corners_sub_pix(image, corners, criteria):
    try:
        corners2 = cv2.cornerSubPix(image, corners, (11,11), (-1,-1), criteria)
        return corners2
    except cv2.error as e:
        print("Error in cornerSubPix:", e)
        return corners
    

def find_external_corners(corners, image, chessboard_size):
    
    top_right_index = 0
    bottom_right_index = chessboard_size[0] - 1
    top_left_index = (chessboard_size[1] - 1) * chessboard_size[0]
    bottom_left_index = -1
       
    top_right = corners[top_right_index][0]
    bottom_right = corners[bottom_right_index][0]
    top_left = corners[top_left_index][0]
    bottom_left = corners[bottom_left_index][0]
    
    top_left_right = corners[top_left_index-chessboard_size[0]][0]
    top_left_bellow = corners[top_left_index+1][0]

    top_right_left = corners[top_right_index+chessboard_size[0]][0]
    top_right_bellow = corners[top_right_index+1][0]
    
    bottom_left_right = corners[bottom_left_index-chessboard_size[0]][0]
    bottom_left_above = corners[bottom_left_index-1][0]

    bottom_right_left = corners[bottom_right_index+chessboard_size[0]][0]
    bottom_right_above = corners[bottom_right_index-1][0]

    Outsider_corners.top_left = (top_left[0] - (top_left_right[0] - top_left[0]+(top_left_bellow[0] - top_left[0])), top_left[1] - (top_left_bellow[1] - top_left[1]) + (top_left_right[1] - top_left[1]))
    Outsider_corners.top_right = (top_right[0] + (-top_right_left[0] + top_right[0]) + (-top_right_bellow[0] + top_right[0]), top_right[1] - (top_right_bellow[1] - top_right[1]) - (top_right_left[1] - top_right[1]))
    Outsider_corners.bottom_left = (bottom_left[0] - (+bottom_left_right[0] - bottom_left[0]) - (+bottom_left_above[0] - bottom_left[0]), bottom_left[1] + (-bottom_left_above[1] + bottom_left[1]) + (-bottom_left_right[1] + bottom_left[1]))
    Outsider_corners.bottom_right = (bottom_right[0] + (-bottom_right_left[0] + bottom_right[0]) + (-bottom_right_above[0] + bottom_right[0]), bottom_right[1] + (-bottom_right_above[1] + bottom_right[1])+ (-bottom_right_left[1] + bottom_right[1]))

    #print("BR: ", bottom_right_above, bottom_right_left, bottom_right, "res: ", Outsider_corners.bottom_right)
    #TODO comment
    cv2.circle(image, (int(Outsider_corners.top_left[0]), int(Outsider_corners.top_left[1])), 1, (0, 255, 255), thickness=2)
    cv2.circle(image, (int(Outsider_corners.top_right[0]), int(Outsider_corners.top_right[1])), 1, (0, 255, 255), thickness=2)
    cv2.circle(image, (int(Outsider_corners.bottom_left[0]), int(Outsider_corners.bottom_left[1])), 1, (0, 255, 255), thickness=2)
    cv2.circle(image, (int(Outsider_corners.bottom_right[0]), int(Outsider_corners.bottom_right[1])), 1, (0, 255, 255), thickness=2)

    image_height, image_width = image.shape[:2]

    if (
        Outsider_corners.top_left[0] < 0 or Outsider_corners.top_left[1] < 0 or
        Outsider_corners.top_right[0] >= image_width or Outsider_corners.top_right[1] < 0 or
        Outsider_corners.bottom_left[0] < 0 or Outsider_corners.bottom_left[1] >= image_height or
        Outsider_corners.bottom_right[0] >= image_width or Outsider_corners.bottom_right[1] >= image_height
    ):
         a= 2
        #Outsider_corners.calculated = False
        #print("The points are outside image.")
    else:
        Outsider_corners.calculated = True
        #print("The points are inside image.")

    return image

def detect_corners_automatically(gray, img, chessboard_size, number_corners=63, threshold= 0.05, min_ec_distance=20):
    print("Detecting corners automatically...")
  
    corners_aux = [Outsider_corners.top_left, Outsider_corners.top_right, Outsider_corners.bottom_right, Outsider_corners.bottom_left]
    Outsider_corners.calculated = False
    print("Making interpolation...")

    corners_np, image = interpolate(img,corners_aux,chessboard_size)

    corners_np = np.array(corners_np, dtype=np.float32).reshape(-1, 1, 2)
    return True, corners_np, img  

def find_and_draw_chessboard_corners(gray, image, chessboard_size, criteria, interval, do_manual, skip_manual):
    """
    Find and draw chessboard corners on the image.


    :param gray: The input image in gray format.
    :param image: The input image.    
    :param chessboard_size: The size of the chessboard.
    :param criteria: Criteria for corner refinement.
    :param interval: Interval of the frames received.
    :param do_manual: Ensures finding the corners manually.
    :para skip_manual: Skip to test.
    :return: The refined corners.
    """
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    #print(corners.shape, "first shape")
    if ret and not do_manual:
        print("Chessboard corners found.")

        corners2 = corners_sub_pix(gray,corners,criteria)          
        image = find_external_corners(corners2, image, chessboard_size)

        cv2.drawChessboardCorners(image, chessboard_size, corners2, ret)
        see_window("Detected corners automatically", image)
        return corners2, image
    elif do_manual or not ret:
        if skip_manual:
            print("Not detected but skip.")
            return None, None
        corners = []
        if do_manual or not Outsider_corners.calculated or interval >= 30: #if the interval is too high it wont work well the automatic corners

            print("Chessboard corners not found. Click on four corners.")

            see_window("Click the four corners please.", image)
            cv2.setMouseCallback('Click the four corners please.', click_event, (corners, image))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            corners, image = interpolate(image, corners, chessboard_size)
            if corners is None or image is None:
                return None, None
            corners2 = corners_sub_pix(gray,corners,criteria) 
            print("Corners found after intrapolation and manually selected corners.")           
            #see_window("Result with Interpolation", image)
            return corners2, image
        else:
            done, corners, image = detect_corners_automatically(gray, image, chessboard_size)

            corners2 = corners_sub_pix(gray,corners,criteria)            
            see_window("Corners Detected Automatically", image)
            return corners2, image


def method_detect_corners_automatically(image, processed_frame, chessboard_size, contour_points, max_contour, criteria):
    
    epsilon = 0.04 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)

    cv2.drawContours(processed_frame, [approx], -1, (0, 255, 0), 2)

    corners = approx.reshape(-1, 2)

    ordered_corners = order_corners_aux(corners)
    final_corners = obtain_inner_corners(ordered_corners, image, chessboard_size, criteria)
    corners, image = interpolate(image, final_corners, chessboard_size)
    if corners is None or image is None:
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners2 = corners_sub_pix(image,corners,criteria) 
    print("Corners found after intrapolation and manually selected corners.")           
    #see_window("Result with Interpolation", image)
    return corners2, image


def order_corners_aux(corners):
    """
    Order corners to match the interpolation.
    :param corners: Unordered corners
    return: Ordered corners
    """
    corners = np.array(corners)

    # Calculate the centroid of the corners
    centroid = np.mean(corners, axis=0)

    # Calculate the angle of rotation
    angle = np.arctan2(corners[:,1] - centroid[1], corners[:,0] - centroid[0])
    angle = np.degrees(angle)

    # Sort corners based on the angle
    sorted_indexes = np.argsort(angle)

    # Rearrange the corners based on the sorted indexes
    ordered_corners = corners[sorted_indexes]

    # Reorder the corners to match the requested order (top-left, top-right, bottom-right, bottom-left)
    if ordered_corners[1][1] > ordered_corners[2][1]:
        ordered_corners[1], ordered_corners[2] = ordered_corners[2], ordered_corners[1]
    
    shrink_factor = 3
    top_left = ordered_corners[0] + shrink_factor
    top_right = ordered_corners[1] - [shrink_factor, 0]
    bottom_right = ordered_corners[2] - shrink_factor
    bottom_left = ordered_corners[3] + [shrink_factor, 0]

    # Adjust the ordered corners with the new values
    ordered_corners[0] = top_left
    ordered_corners[1] = top_right
    ordered_corners[2] = bottom_right
    ordered_corners[3] = bottom_left
    

    return ordered_corners

def obtain_inner_corners(corners, image, chessboard_size, criteria):

    """
    Obtain inner corners for the interpolation.
    :param corners: Ordered corners.
    :param image: Image for shape.
    return: Ordered corners
    """
    original_corners = corners
    height, width, _ = image.shape


    dst_points = np.array([[0, height - 1], [width - 1, height - 1], [width - 1, 0],[0,0]], dtype="float32")

    matrix = cv2.getPerspectiveTransform(np.array(corners, dtype="float32"), dst_points)

    auxiliar_warped = cv2.warpPerspective(image, matrix, (width, height))
    
    gray = cv2.cvtColor(auxiliar_warped, cv2.COLOR_BGR2GRAY)
    threshold = 150

    """#cv2.circle(gray, (int(top_left[0]), int(top_left[1])), 5, (0, 255, 255), -1)
    cv2.circle(gray, (top_right[0], top_right[1]), 5, (0, 255, 255), -1)
    cv2.circle(gray, bottom_right,5, (244, 0, 255), -1)
    cv2.circle(gray, bottom_left, 5, (244, 234, 255), -1)"""
    
    #gray2 = preprocess_image(auxiliar_warped,True)


    adjusted_corners =aux_points(auxiliar_warped,threshold)
    adjusted_corners = np.array(adjusted_corners)
    corners_np = np.zeros((4, 1, 2), dtype=np.float32)

    for i, corner in enumerate(adjusted_corners):
        corners_np[i, 0, 0] = corner[0]  
        corners_np[i, 0, 1] = corner[1] 

    corners_original_image = cv2.perspectiveTransform(corners_np, np.linalg.inv(matrix))
    corners_original_image = np.array(corners_original_image, dtype=np.float32).reshape(-1, 1, 2)
    print(corners_original_image)
    print("_- orig")
    print(original_corners)
    return corners_original_image

def aux_points(img, threshold):

    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(img2,(25,25),0) 
    ret, binary = cv2.threshold(blur,123,255,cv2.THRESH_BINARY) 
    binary = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    contours = filter_contours(contours)
    for contour in contours:
        cv2.drawContours(img, [contour], -1, (255, 0, 0), 3)
    #see_window("·R", img)

    top_left = (float('inf'), float('inf'))
    top_right = (float('-inf'), float('inf'))
    bottom_right = (float('-inf'), float('-inf'))
    bottom_left = (float('inf'), float('-inf'))
    
    for contour in contours:
        for point in contour:
            x, y = point[0]
            if x + y < top_left[0] + top_left[1]:
                top_left = (x, y)
            if x - y > top_right[0] - top_right[1]:
                top_right = (x, y)
            if x + y > bottom_right[0] + bottom_right[1]:
                bottom_right = (x, y)
            if x - y < bottom_left[0] - bottom_left[1]:
                bottom_left = (x, y)

    cv2.circle(img, top_left, 10, (0, 255, 0), -1)
    cv2.circle(img, top_right, 10, (0, 255, 0), -1)
    cv2.circle(img, bottom_right, 10, (0, 255, 0), -1)
    cv2.circle(img, bottom_left, 10, (0, 255, 0), -1)
    
    # Display the image
    #see_window("Contours with Extreme Points", img)
    corners = []
    # the order of the corners must be modified after the warped image to match the interpolation
    corners.append(bottom_left)
    corners.append(bottom_right)
    corners.append(top_right)
    corners.append(top_left)
    return corners

def filter_contours(contours, min_area= 500, max_area=10000):
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            filtered_contours.append(contour)
    return filtered_contours