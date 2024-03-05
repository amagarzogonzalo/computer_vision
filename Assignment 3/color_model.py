import cv2 as cv
import numpy as np

def clustering(voxel_list, N=4):
    print(voxel_list)
    # we don't consider the height
    voxel_list = np.array(voxel_list).astype(np.float32)[:, [0, 2]] 
    print(voxel_list)

    # define criteria and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,labels,centers=cv.kmeans(voxel_list,N,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

    return labels, centers


def construct_color_model(voxel_list, labels, centers, selected_frame, lookup_table_selected_camera, selected_camera):
    labels = np.ravel(labels)

    #TODO CHECK AND CHANGE
    
    frame = cv.cvtColor(selected_frame, cv.COLOR_BGR2HSV)
    
    color_for_label = []
    pixel_label_list = []
    for label in range(4):
        pixel_list = []
        voxel_indices = np.where(labels == label)[0]
        print("Vsd", voxel_indices)
        voxel_label = []
        for pos in voxel_indices:
            voxel_label.append(voxel_list[pos])
        for voxel in voxel_label:
            print(voxel)
            pixel_i = lookup_table_selected_camera[selected_camera][tuple(voxel)]
            pixel_list.append(pixel_i)

            roi = np.array([frame[y, x] for [x, y] in pixel_list])
            roi = np.float32(roi)
            model = cv.ml.EM_create()
            model.setClustersNumber(4)
            model.trainEM(roi)

            color_for_label.append(model)

        pixel_label_list.append(pixel_list)


    return color_for_label, pixel_label_list


def paint_image(image, pixel_list):
    colors = [
    (0, 0, 255),  # Red
    (0, 255, 0),  # Green
    (255, 0, 0),  # Blue
    (255, 255, 0)  # Yellow
    ] 
    image_aux = image
    for pixels, color in zip(pixel_list, colors):
        for pixel in pixels:
            x, y = pixel
            image_aux[y, x] = color
    cv.imshow('Painted Image', image_aux)
    cv.imshow('Image normal', image)

    cv.waitKey(0)
    cv.destroyAllWindows()


def color_model(voxel_list, frames_cam, lookup_table_selected_camera, selected_camera):

    labels, centers = clustering(voxel_list)

    _, pixel_list  = construct_color_model(voxel_list, labels, centers, frames_cam[selected_camera], lookup_table_selected_camera, selected_camera)
    paint_image(frames_cam[selected_camera], pixel_list)
    


