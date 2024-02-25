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

def detect_corners_for_extrinsic(gray, img, chessboard_size, number_corners=48, threshold= 0.005, min_ec_distance=5):
    corners = cv2.goodFeaturesToTrack(gray,number_corners,threshold, min_ec_distance, useHarrisDetector=True, k=0.005)
    corners = np.int0(corners)
    #see_window("Aux", img)
    return corners

def draw_corners(image, corners):
    corners = corners.reshape((-1, 2))
    #print(corners)

    for point in corners:
        cv2.circle(image, (int(point[0]), int(point[1])), 1, (255, 0, 0))   
    draw_new_corners(image, image, corners)
    return image 

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

#----------------
        
def check_for_bigger_line(lines):
    bigger_lines = [] 
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line1 = lines[i]
            line2 = lines[j]
            # Check if the lines are collinear
            #if np.cross(np.array(line1[1]) - np.array(line1[0]), np.array(line2[1]) - np.array(line1[0])) == 0:
            if (collinear(line1[0][0], line1[0][1], line1[1][0], line1[1][1], line2[1][0], line2[1][1]) or collinear(line1[0][0], line1[0][1], line1[1][0], line1[1][1], line2[0][0], line2[0][1])) and line1[0][0] != line2[0][0] and line1[0][0] != line2[1][0] and line1[1][0] != line2[1][0] and line1[1][0] != line2[0][0]:
                # Lines are collinear, can potentially form a bigger line
                #approximate
                print(line1[0], ".", line1[1], "||", line2[0], ".", line2[1])
                small_point = min(line1[0], line1[1], line2[0], line2[1])
                big_point = max(line1[0], line1[1], line2[0], line2[1])
                # Create the bigger line using the smallest and largest points
                bigger_line = (small_point, big_point)                
                print("Lines", i, "and", j, "are collinear and can potentially form a bigger line.")
                bigger_lines.append(bigger_line)
                #return bigger_lines
    return bigger_lines


def collinear(x1, y1, x2, y2, x3, y3):

    """ Calculation the area of  
        triangle. We have skipped 
        multiplication with 0.5 to
        avoid floating point computations """
    a = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    if a == 0:
        return True
    else:
        return False

def draw_new_corners(image, gray, corners):
    corners = np.array(corners, dtype='int32')
    lines = []
    for i in range(len(corners)):
        x1, y1 = corners[i].ravel()

        # Calculate distances to all other corners in 2D space
        distances = np.linalg.norm(corners - corners[i], axis=1)

        # Exclude the current corner's distance (distance to itself)
        distances[i] = 0

        # Find the indices of the two nearest corners
        nearest_indices = np.argsort(distances)[1:2]

        # Draw lines to the two nearest corners
        for j in nearest_indices:
            x2, y2 = corners[j].ravel()
            lines.append(((x1, y1), (x2, y2)))

            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
            #cv2.line(gray, (x1, y1), (x2, y2), 255, 1)


    bigger_lines = check_for_bigger_line(lines)
    print(len(bigger_lines))
    if len(bigger_lines)>0:
        print(bigger_lines)
        for bigger_line in bigger_lines:
            cv2.line(image, bigger_line[0], bigger_line[1], (255, 0, 255), 2)
    return image, gray 

def extract_corners (corners, image, chessboard_size, gray, number_corners):
    print("extract corners")
    corners = corners.reshape(number_corners, 2)
    #print(corners.shape, corners) 
    threshold = 15
    # idea-> find top corners and use it for interpolation, find topleft topright bottomleft bottomright

    #interpolate(image,top_four_corners, chessboard_size)
    draw_corners(image, gray, corners)