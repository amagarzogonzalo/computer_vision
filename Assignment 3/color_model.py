import cv2 as cv
import numpy as np


def color_model(voxel_list):
    clustering(voxel_list)


def clustering(voxel_list, centers=4):
    print(voxel_list)
    voxel_list = np.array(voxel_list).astype(np.float32)
    print(voxel_list)

    # define criteria and apply kmeans()
    
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    #ret,label,center=cv.kmeans(,2,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

