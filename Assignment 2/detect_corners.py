import cv2
import numpy as np
from interpolate import interpolate, getEquidistantPoints
from auxiliar import see_window, click_event

def corners_sub_pix(image, corners, criteria):
    try:
        corners2 = cv2.cornerSubPix(image, corners, (11,11), (-1,-1), criteria)
    except cv2.error as e:
        print("Error in cornerSubPix:", e)
        corners2 = corners   
    return corners2


def check_for_bigger_line(lines):
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line1 = lines[i]
            line2 = lines[j]
            if line1[0] == line2[0] or line1[0] == line2[1]:
                if np.cross(np.array(line1[1]) - np.array(line1[0]), np.array(line2[1]) - np.array(line1[0])) == 0:
                    print("Lines", i, "and", j, "are collinear and can potentially form a bigger line.")

def draw_corners(image, gray, corners):
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

            #cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
            #cv2.line(gray, (x1, y1), (x2, y2), 255, 1)
        
        cv2.circle(image,(x1,y1),3,255,-1)
        cv2.circle(gray,(x1,y1),3,255,-1)


    bigger_lines = check_for_bigger_line(lines)
    if len(bigger_lines)>0:
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


def detect_corners_automatically(gray, img, chessboard_size, number_corners=63, threshold= 0.005, min_ec_distance=20):
    corners = cv2.goodFeaturesToTrack(gray,number_corners,threshold, min_ec_distance)
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
