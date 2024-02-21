import cv2
import numpy as np
from interpolate import interpolate, getEquidistantPoints
from auxiliar import see_window, click_event
from scipy.spatial.distance import cdist

class Outsider_corners:
    top_left = (0,0)
    top_right = (0,0)
    bottom_left = (0,0)
    bottom_right = (0,0)

def corners_sub_pix(image, corners, criteria):
    return corners
    try:
        corners2 = cv2.cornerSubPix(image, corners, (11,11), (-1,-1), criteria)
    except cv2.error as e:
        print("Error in cornerSubPix:", e)
        corners2 = corners   
    return corners2

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

    Outsider_corners.top_left = ((top_left[0]-(top_left_right[0]-top_left[0])),   (top_left[1]-(top_left_bellow[1]-top_left[1])))
    Outsider_corners.top_right = ((top_right[0]+(-top_right_left[0]+top_right[0])),   (top_right[1]-(top_right_bellow[1]-top_right[1])))
    Outsider_corners.bottom_left = ((bottom_left[0]-(bottom_left_right[0]-bottom_left[0])),   (bottom_left[1]+(-bottom_left_above[1]+bottom_left[1])))
    Outsider_corners.bottom_right = ((bottom_right[0]+(-bottom_right_left[0]+bottom_right[0])),   (bottom_right[1]+(-bottom_right_above[1]+bottom_right[1])))



    cv2.circle(image, (int(Outsider_corners.top_left[0]), int(Outsider_corners.top_left[1])), 1, (0, 255, 255), thickness=2)
    cv2.circle(image, (int(Outsider_corners.top_right[0]), int(Outsider_corners.top_right[1])), 1, (0, 255, 255), thickness=2)
    cv2.circle(image, (int(Outsider_corners.bottom_left[0]), int(Outsider_corners.bottom_left[1])), 1, (0, 255, 255), thickness=2)
    cv2.circle(image, (int(Outsider_corners.bottom_right[0]), int(Outsider_corners.bottom_right[1])), 1, (0, 255, 255), thickness=2)


    #for point in closest_top_left:
     #   cv2.circle(image, (int(point[0]), int(point[1])), 7, (0, 0, 0), -1)


    # white yellow green red
    print(top_left)
    cv2.circle(image,(int(top_left[0]), int(top_left[1])),7,(255,255,255),-1)
    cv2.circle(image,(int(top_left_right[0]), int(top_left_right[1])),5,(255,255,255),-1)
    cv2.circle(image,(int(top_left_bellow[0]), int(top_left_bellow[1])),9,(255,255,255),-1)

    cv2.circle(image,(int(bottom_right[0]), int(bottom_right[1])),7,(0,255,255),-1)
    cv2.circle(image,(int(top_right[0]), int(top_right[1])),11,(0,255,0),-1)
    cv2.circle(image,(int(bottom_left[0]), int(bottom_left[1])),7,(0,0,255),-1)
    i = 1
    for corner in corners:
    # Extrae las coordenadas del punto
        x, y = corner[0]


        # Dibuja un círculo naranja alrededor del punto
        # cv2.circle(img, center, radius, color, thickness)
        # Aumenta ligeramente el radio en cada iteración
        radius = i   # Incremento de radio
        cv2.circle(image, (int(x), int(y)), radius, (0, 165, 255), thickness=2)
        i+= 1
   
    return image

def draw_corners(image, gray, corners):
    print("MANUAL")
    corners = np.array(corners, dtype='int32')
 
    for i in range(len(corners)):
        x1, y1 = corners[i].ravel()
       
        
        cv2.circle(image,(x1,y1),3,255,-1)
        cv2.circle(gray,(x1,y1),3,255,-1)


    return image, gray 

def extract_corners (corners, image, chessboard_size, gray, number_corners):
    corners = corners.reshape(number_corners, 2)
    #print(corners.shape, corners) 
    threshold = 15
    # idea-> find top corners and use it for interpolation, find topleft topright bottomleft bottomright

    #interpolate(image,top_four_corners, chessboard_size)
    draw_corners(image, gray, corners)


def detect_corners_automatically(gray, img, chessboard_size, number_corners=63, threshold= 0.05, min_ec_distance=20):
    corners = cv2.goodFeaturesToTrack(gray,number_corners,threshold, min_ec_distance, useHarrisDetector=True, k=0.005)
    corners_draw = np.int32(corners)
    corners_np = np.float32(corners)
    extract_corners(corners_np, img, chessboard_size, gray, len(corners))

    see_window("AUXILIAR", gray)
    corners_np = np.array(corners_np, dtype=np.float32).reshape(-1, 1, 2)
    #img, gray = draw_corners(img, gray, corners_draw)
    return True, corners_np, img  

def find_and_draw_chessboard_corners(gray, image, chessboard_size, criteria):
    """
    Find and draw chessboard corners on the image.


    :param gray: The input image in gray format.
    :param image: The input image.    
    :param chessboard_size: The size of the chessboard.
    :param criteria: Criteria for corner refinement.
    :return: The refined corners.
    """
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        print("Chessboard corners found.")
        corners2 = corners_sub_pix(gray,corners,criteria)          
        cv2.drawChessboardCorners(image, chessboard_size, corners2, ret)
        
        image = find_external_corners(corners, image, chessboard_size)
        see_window("Detected corners automatically", image)
        return corners2, image
    else:
        
        done, corners, image = detect_corners_automatically(gray, image, chessboard_size)
        corners = []
        if not done:
            print("Chessboard corners not found. Click on four corners.")
            see_window("Image", image)
            cv2.setMouseCallback('Image', click_event,  (corners, image))
            cv2.waitKey(0)
            corners, image = interpolate(image, corners, chessboard_size)
            if corners is None or image is None:
                return None
            corners2 = corners_sub_pix(gray,corners,criteria)            
            see_window("Result with Interpolation", image)
            return corners2, image
        else:
            corners2 = corners_sub_pix(gray,corners,criteria)            
            see_window("Corners Detected Automatically", image)
            return corners2, image


