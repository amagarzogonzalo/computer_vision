import cv2

def click_event(event, x, y, flags, params): 
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
        # displaying the coordinates 
        # on the Shell 
        print(x, ' ', y) 
       
def corner_detection (img):

    # Find the chess board corners

    # if chessboard corners are detected
    if ret == True:
    
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,6), corners,ret)
        cv2.imshow('Chessboard',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
  


if __name__=="__main__": 
  
    # reading the image 
    img = cv2.imread('lena.jpg', 1) 
    # convert the input image to a grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # displaying the image 
    cv2.imshow('image', img) 
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
    if ret == False: 
        # setting mouse handler for the image 
        # and calling the click_event() function 
        cv2.setMouseCallback('image', click_event) 
    
        # wait for a key to be pressed to exit 
        cv2.waitKey(0) 
    
        # close the window 
        cv2.destroyAllWindows() 

    else :
    
  