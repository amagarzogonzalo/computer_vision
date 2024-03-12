import cv2 as cv
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
from scipy.interpolate import interp1d



def read_all_frames(duration_video_secs= 53, total_frames_camera_i=10):
    # frames for all cameras
    frames = []
    # interval between frames: 24 seconds and we are looking for 10 frames
    interval = duration_video_secs * 1000 / total_frames_camera_i
    for i in range(4):
        # frames for each camera
        aux_frames = [] 
        path_name = f'data/cam{i+1}/'

        camera_handle = cv.VideoCapture(path_name + '/video.avi')

        j = 0
        while len(aux_frames) < total_frames_camera_i:
            # read frames for each camera and store it in auxframes = frames of camera i
            camera_handle.set(cv.CAP_PROP_POS_MSEC, j * interval)

            _, frame = camera_handle.read()
            if frame is None:
                break
            aux_frames.append(frame)
            j+=1
        camera_handle.release()
        frames.append(aux_frames)


    return frames

def remove_outliers_iqr(voxel_list):
    Q1 = np.percentile(voxel_list, 25, axis=0)
    Q3 = np.percentile(voxel_list, 75, axis=0)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Keep voxels within the bounds for all dimensions
    inlier_mask = np.all((voxel_list >= lower_bound) & (voxel_list <= upper_bound), axis=1)
    voxels_filtered = voxel_list[inlier_mask]

    return voxels_filtered, None



def clustering(voxel_list, N=4, debug = False):
    voxel_list = np.array(voxel_list).astype(np.float32)[:, [0, 2]]

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    ret, labels, centers = cv.kmeans(voxel_list, N, None, criteria, 10, cv.KMEANS_PP_CENTERS)

    if debug:
        unique, counts = np.unique(labels, return_counts=True)

        with open('labels.txt', 'w') as f:
            f.write("Frequency of each label:\n")
            f.write(f"Label {unique}: {counts}\n\n")
            for label in labels:
                f.write(f"{label}\n")


    return labels, centers

def get_colour (label):
    #TODO: is it good? 
    if label == 0:
        return [0,0,255]
    elif label == 1:
        return [0,255,0]
    elif label == 2:
        return [255,0,0]
    elif label == 3:
        return [255,255,0]


def construct_color_model(voxel_list, labels, centers, lookup_table_every_camera, frames_cam):
    color_models = []
    pixel_label_list_cameras = []

    for i in range(4):
        print(f"Camera: {i+1}")
        labels = np.ravel(labels)
        frame = cv.cvtColor(frames_cam[i], cv.COLOR_BGR2HSV)
        

        pixel_label_list = []
        new_voxel_list = []
        new_colors = []
        colors = []
        
        for label in range(len(np.unique(labels))):
            print(f"Processing label {label}.")
            # Filter voxels by label
            voxels_person = np.array(voxel_list)[labels == label]

            # Calculate the 't-shirt' and 'head' cutoffs
            tshirt = np.mean(voxels_person[:, 1])
            head = np.max(voxels_person[:, 1])
            # Create ROI based on 'tshirt' and 'head' values
            voxel_roi = (voxels_person[:, 1] > tshirt) & (voxels_person[:, 1] < 3 / 4 * head)
            voxels_person_roi = voxels_person[voxel_roi]

            pixel_list = []
            for voxel in voxels_person_roi:
                #pixel = lookup_table_selected_camera.get(tuple(voxel), None)
                pixel = lookup_table_every_camera[i+1].get(tuple(voxel), None)

                if pixel:
                    pixel_list.append(pixel)
                    new_voxel_list.append(voxel)
            
                    new_color = get_colour(label)
                    new_colors.append(new_color)
                    #new_colors.append(np.array(new_color,dtype= np.float32))
                    
            print(f"Label {label}: Found {len(pixel_list)} corresponding pixels in lookup table.")

            if len(pixel_list) > 0:
                # Convert list of (x, y) pixel coordinates to ROI for GMM
                roi = np.array([frame[y, x] for x, y in pixel_list])
                roi = np.float32(roi)

                # Create and train the GMM (EM) model
                model = GaussianMixture(n_components=4, covariance_type='full')
                model.fit(roi[:, :2])  # Fit to the H and S channels
                print(f"Successfully trained GMM model for label {label}.")

                # Store the GMM model
                colors.append(model)


                #TODO: Do not know yet if it is useful
                num_components = model.n_components
                mean_color_cluster = []
                for component in range(num_components):
                    mean_color = model.means_[component]
                    mean_color_three = np.append(mean_color,255) # It has 2 channels
                    mean_color_cluster.append(mean_color_three)

                    
            else:
                print(f"Not enough pixels to train GMM model for label {label}, skipping.")
                colors.append(None)

            pixel_label_list.append(pixel_list)

        color_models.append(colors)
        pixel_label_list_cameras.append(pixel_label_list)



    return new_voxel_list, new_colors, color_models, pixel_label_list_cameras


    """for i in range(len(new_colors)):
            color = new_colors[i]
            if color == [0,0,255]:
                new_colors[i] = mean_color_cluster[0]
            elif color == [0,255,0]:
                new_colors[i] = mean_color_cluster[1]

            elif color == [255,0,0]:
                new_colors[i] = mean_color_cluster[2]

            elif color == [255,255,0]:
                new_colors[i] = mean_color_cluster[3]
    """


def paint_image(image, pixel_list):
    colors = [
        (0, 0, 255),  # Red
        (0, 255, 0),  # Green
        (255, 0, 0),  # Blue
        (255, 255, 0)  # Yellow
    ]
    image_aux = image.copy()
    for pixels, color in zip(pixel_list, colors):
        for pixel in pixels:
            x, y = pixel
            image_aux[y, x] = color
    cv.imshow('Painted Image in offline', image_aux)
    cv.imshow('Image normal', image)

    cv.waitKey(0)





def color_model(voxel_list, frames_cam, lookup_table_selected_camera, selected_camera, lookup_table_every_camera):
    # Remove outliers from the entire voxel_list before clustering
    voxels_filtered, _ = remove_outliers_iqr(np.array(voxel_list))

    labels, centers = clustering(voxels_filtered)

    new_voxel_list, new_colors, color_models, pixel_label_list_cameras = construct_color_model(voxels_filtered, labels, centers, lookup_table_every_camera, frames_cam)

    selected_camera_aux = 1
    paint_image(frames_cam[selected_camera_aux], pixel_label_list_cameras[selected_camera_aux])
    return new_voxel_list, new_colors, color_models



def online_phase(colors_model, voxel_list, frames_cam, lookup_table_every_camera, curr_time, debug= True) :
    print("---Online phase---")
    voxels_filtered, _ = remove_outliers_iqr(np.array(voxel_list))

    labels, centers = clustering(voxels_filtered)

    frames = read_all_frames()

    probabilities_labels = []
    calculated_labes = [None, None, None, None]
    new_voxel_list = []
    number_voxels_label = [None, None, None, None]
    new_colors = []
     
   
    for i in range(4): 
        frame = cv.cvtColor(frames[i][curr_time], cv.COLOR_BGR2HSV)
        labels = np.ravel(labels)

        for label in range(len(np.unique(labels))):
            # voxels that have this label
            label_voxel = []

            voxels_person = np.array(voxels_filtered)[labels == label]

            # Calculate the 't-shirt' and 'head' cutoffs
            tshirt = np.mean(voxels_person[:, 1])
            head = np.max(voxels_person[:, 1])
            # Create ROI based on 'tshirt' and 'head' values
            voxel_roi = (voxels_person[:, 1] > tshirt) & (voxels_person[:, 1] < 3 / 4 * head)
            voxels_person_roi = voxels_person[voxel_roi]

            pixel_list = []
            auxiliar_number_voxels_label = 0
            for voxel in voxels_person_roi:
                #pixel = lookup_table_selected_camera.get(tuple(voxel), None)
                pixel = lookup_table_every_camera[i+1].get(tuple(voxel), None)

                if pixel:
                    pixel_list.append(pixel)
                    label_voxel.append(voxel)
                    # we only save the voxels once
                    if i == 0:
                        new_voxel_list.append(voxel)
                        auxiliar_number_voxels_label += 1
                        #TODO: delete, just beffore obtaining real colors
                        new_color = get_colour(label)
                        new_colors.append(new_color)



            number_voxels_label[i] = auxiliar_number_voxels_label
            # first part of online - match colour with cluster
            probabilities = []
            # for each cluster:  len of color models of camera i
            for j in range(len(colors_model[i])):
                roi = np.array([frame[y, x][:2] for x, y in pixel_list])
                roi = np.float32(roi)
                total_prob = 0
                for sample in roi:
                    #print(sample)
                    # we have 1d array and we need a 2d array
                    sample_2d = sample.reshape(1, -1)
                    #print(sample_2d)
                    prob = colors_model[i][j].predict_proba(sample_2d)
                    total_prob += prob
                probabilities.append(total_prob)
        probabilities_labels.append(probabilities)
        # match person and colour

        print(np.array(probabilities_labels))
        _, labels = optimize.linear_sum_assignment(np.array(probabilities_labels))
        calculated_labes[i] = labels.tolist()
        print("Camera i:  ........................................................ ",calculated_labes)
    if debug:

        with open('debug.txt', 'w') as f:
            
            f.write("Probaiblities:\n")
            for p in probabilities_labels:
                f.write("camera i::\n")

                for a in p:
                    f.write(f"{a} ")
                f.write("\n")

            f.write("Calculated:\n")
            for p in calculated_labes:
                f.write("camera i::\n")

                for a in p:
                    f.write(f"{a} ")
                f.write("\n")

    # Assign colour
                
    cluster_asssigned = 0
    while cluster_asssigned < 4:


        cluster_asssigned+=1

    return new_voxel_list, new_colors


    
            

