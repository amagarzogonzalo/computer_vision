import cv2
import os
from threading import Thread
  #https://rdmilligan.wordpress.com/2015/06/28/opencv-camera-calibration-and-pose-estimation-using-python/




def webcam_mode():
    """Webcam mode that allows us to saving images by pressing spacebar,
      and terminates by pressing enter (pressing any other key continues displaying frames until termination)."""
    cam=cv2.VideoCapture(0)
    i=1
    while True:
        hasframe,frame=cam.read()
        if hasframe==False:
            break    
        cv2.imshow('Output',frame)
        if cv2.waitKey(1)==13:
            break
        elif cv2.waitKey(32)==ord(' '):
            filename = os.path.join("webcam", str(i) + '.jpg')

            cv2.imwrite(filename,frame)
            i+=1
            print('Captured')
        else:
            continue
    cv2.destroyAllWindows()
    cam.release()
