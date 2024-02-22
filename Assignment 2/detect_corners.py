import cv2
import numpy as np
from interpolate import interpolate, getEquidistantPoints
from auxiliar import see_window, click_event
from scipy.spatial.distance import cdist
from scipy.spatial import distance


class Outsider_corners:
    calculated = False
    distance_threshold = 0
    top_left = (0,0)
    top_right = (0,0)
    bottom_left = (0,0)
    bottom_right = (0,0)


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

    """Outsider_corners.distance_threshold = 1.5*(top_left[1]-top_left_bellow[1])
    Outsider_corners.top_left = ((top_left[0]-(top_left_right[0]-top_left[0])),   (top_left[1]-(top_left_bellow[1]-top_left[1])))
    Outsider_corners.top_right = ((top_right[0]+(-top_right_left[0]+top_right[0])),   (top_right[1]-(top_right_bellow[1]-top_right[1])))
    Outsider_corners.bottom_left = ((bottom_left[0]-(bottom_left_right[0]-bottom_left[0])),   (bottom_left[1]+(-bottom_left_above[1]+bottom_left[1])))
    Outsider_corners.bottom_right = ((bottom_right[0]+(-bottom_right_left[0]+bottom_right[0])+(-bottom_right_above[0]+bottom_right[0]),   (bottom_right[1]+(-bottom_right_above[1]+bottom_right[1])))
    """
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
    corners = cv2.goodFeaturesToTrack(gray,number_corners,threshold, min_ec_distance, useHarrisDetector=True, k=0.005)
    corners_draw = np.int32(corners)
    print("Detecting corners automatically...")
    corners_np = np.float32(corners)
    #extract_corners(corners_np, img, chessboard_size, gray, len(corners))
    corners_aux = [Outsider_corners.top_left, Outsider_corners.top_right, Outsider_corners.bottom_right, Outsider_corners.bottom_left]
    Outsider_corners.calculated = False
    print("Making interpolation...")

    corners_np, image = interpolate(img,corners_aux,chessboard_size)
    #find_external_corners(corners_np,image,chessboard_size)

    #see_window("Image with points detected from previous frame.", image)
    corners_np = np.array(corners_np, dtype=np.float32).reshape(-1, 1, 2)
    #img, gray = draw_corners(img, gray, corners_draw)
    return True, corners_np, img  

def find_and_draw_chessboard_corners(gray, image, chessboard_size, criteria, interval):
    """
    Find and draw chessboard corners on the image.


    :param gray: The input image in gray format.
    :param image: The input image.    
    :param chessboard_size: The size of the chessboard.
    :param criteria: Criteria for corner refinement.
    :param interval: Interval of the frames received.
    :return: The refined corners.
    """
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    #print(corners.shape, "first shape")

    if ret:
        print("Chessboard corners found.")

        corners2 = corners_sub_pix(gray,corners,criteria)          
        image = find_external_corners(corners2, image, chessboard_size)

        cv2.drawChessboardCorners(image, chessboard_size, corners2, ret)
        see_window("Detected corners automatically", image)
        return corners2, image
    else:
        skip_manual = False # just for testing
        if skip_manual:
            return None, None
        corners = []
        if not Outsider_corners.calculated or interval >= 30: #if the interval is too high it wont work well the automatic corners

            print("Chessboard corners not found. Click on four corners.")

            see_window("Click the four corners please.", image)
            cv2.setMouseCallback('Click the four corners please.', click_event, (corners, image))
            cv2.waitKey(0)
            corners, image = interpolate(image, corners, chessboard_size)
            if corners is None or image is None:
                return None
            corners2 = corners_sub_pix(gray,corners,criteria) 
            print("Corners found after intrapolation and manually selected corners.")           
            see_window("Result with Interpolation", image)
            return corners2, image
        else:
            done, corners, image = detect_corners_automatically(gray, image, chessboard_size)

            corners2 = corners_sub_pix(gray,corners,criteria)            
            see_window("Corners Detected Automatically", image)
            return corners2, image


"""
def draw_corners(image, gray, corners):
    #print("MANUAL")
    corners = np.array(corners, dtype='int32')
    if Outsider_corners.calculated is True:
        cv2.circle(image, (int(Outsider_corners.top_left[0]), int(Outsider_corners.top_left[1])), 1, (0, 255, 255), thickness=2)
        cv2.circle(image, (int(Outsider_corners.top_right[0]), int(Outsider_corners.top_right[1])), 1, (0, 255, 255), thickness=2)
        cv2.circle(image, (int(Outsider_corners.bottom_left[0]), int(Outsider_corners.bottom_left[1])), 1, (0, 255, 255), thickness=2)
        cv2.circle(image, (int(Outsider_corners.bottom_right[0]), int(Outsider_corners.bottom_right[1])), 1, (0, 255, 255), thickness=2)
 
    for i in range(len(corners)):
        x1, y1 = corners[i].ravel()
       
        
        cv2.circle(image,(x1,y1),3,255,-1)
        cv2.circle(gray,(x1,y1),3,255,-1)


    return image, gray 

def extract_corners (corners, image, chessboard_size, gray, number_corners):
    
    #Obtain the nearest point according to Harris.
    
    corners = corners.reshape(number_corners, 2)

    distances = np.zeros((number_corners, 4))
    for i, corner in enumerate(corners):
        distances[i, 0] = np.linalg.norm(corner - Outsider_corners.top_left)
        distances[i, 1] = np.linalg.norm(corner - Outsider_corners.top_right)
        distances[i, 2] = np.linalg.norm(corner - Outsider_corners.bottom_left)
        distances[i, 3] = np.linalg.norm(corner - Outsider_corners.bottom_right)

    closest_corners_indices = np.argmin(distances, axis=0)
    if Outsider_corners.calculated is False:
        #np.any(np.min(distances, axis=1) > Outsider_corners.distance_threshold) or np.any(np.min(distances, axis=0) > Outsider_corners.distance_threshold):
        print("No found 4 corners")
    else:
        top_left = corners[closest_corners_indices[0]]
        top_right = corners[closest_corners_indices[1]]
        bottom_left = corners[closest_corners_indices[2]]
        bottom_right = corners[closest_corners_indices[3]]
        cv2.circle(image, (int(top_left[0]), int(top_left[1])), 5, (0, 0, 255), thickness=2)
        cv2.circle(image, (int(top_right[0]), int(top_right[1])), 5, (0, 0, 255), thickness=2)
        cv2.circle(image, (int(bottom_left[0]), int(bottom_left[1])), 5, (0, 0, 255), thickness=2)
        cv2.circle(image, (int(bottom_right[0]), int(bottom_right[1])), 5, (0, 0, 255), thickness=2)
        draw_corners(image, gray, corners)

"""