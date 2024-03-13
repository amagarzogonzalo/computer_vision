import cv2 as cv
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
from scipy.interpolate import interp1d


trajectory_data = []


def read_all_frames(duration_video_secs=53, total_frames_camera_i=10):
    # frames for all cameras
    frames = []
    # interval between frames: 24 seconds and we are looking for 10 frames
    interval = duration_video_secs * 1000 / total_frames_camera_i
    for i in range(4):
        # frames for each camera
        aux_frames = []
        path_name = f'data/cam{i + 1}/'

        camera_handle = cv.VideoCapture(path_name + '/video.avi')

        j = 0
        while len(aux_frames) < total_frames_camera_i:
            # read frames for each camera and store it in auxframes = frames of camera i
            camera_handle.set(cv.CAP_PROP_POS_MSEC, j * interval)

            _, frame = camera_handle.read()
            if frame is None:
                break
            aux_frames.append(frame)
            j += 1
        camera_handle.release()
        frames.append(aux_frames)

    return frames


def remove_outliers_iqr(voxel_list):
    # Calculate the first quartile (Q1), third quartile (Q3), and interquartile range (IQR) 
    Q1 = np.percentile(voxel_list, 25, axis=0)
    Q3 = np.percentile(voxel_list, 75, axis=0)
    IQR = Q3 - Q1

    # Calculate the lower and upper bounds for outlier detection
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Create a mask to identify inliers within the bounds for all dimensions
    inlier_mask = np.all((voxel_list >= lower_bound) & (voxel_list <= upper_bound), axis=1)
    
    # Filter out outliers based on mask
    voxels_filtered = voxel_list[inlier_mask]

    return voxels_filtered, None


def clustering(voxel_list, N=4, debug=True):
    # Cluster method performing k-means from opencv
    voxel_list = np.array(voxel_list).astype(np.float32)[:, [0, 2]]

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    ret, labels, centers = cv.kmeans(voxel_list, N, None, criteria, 10, cv.KMEANS_PP_CENTERS)

    if debug:
        # debug method to check the label distribution
        unique, counts = np.unique(labels, return_counts=True)

        with open('labels.txt', 'w') as f:
            f.write("Frequency of each label:\n")
            f.write(f"Label {unique}: {counts}\n\n")
            for label in labels:
                f.write(f"{label}\n")

    return labels, centers


def get_colour(label):
    # Assign color based on label
    if label == 0:
        return [0, 0, 255]
    elif label == 1:
        return [0, 255, 0]
    elif label == 2:
        return [255, 0, 0]
    elif label == 3:
        return [255, 255, 0]


def construct_color_model(voxel_list, labels, centers, lookup_table_every_camera, frames_cam):
    color_models = []
    pixel_label_list_cameras = []
    
    # iterate over cameras
    for i in range(4):
        print(f"Camera: {i + 1}")
        labels = np.ravel(labels)
        frame = cv.cvtColor(frames_cam[i], cv.COLOR_BGR2HSV)

        pixel_label_list = []
        new_voxel_list = []
        new_colors = []
        colors = []

        # iterate over labels
        for label in range(len(np.unique(labels))):
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
                # pixel = lookup_table_selected_camera.get(tuple(voxel), None)
                pixel = lookup_table_every_camera[i + 1].get(tuple(voxel), None)

                if pixel:
                    pixel_list.append(pixel)
                    new_voxel_list.append(voxel)

                    new_color = get_colour(label)
                    new_colors.append(new_color)
                    # new_colors.append(np.array(new_color,dtype= np.float32))

            print(f"Label {label}: Found {len(pixel_list)} corresponding pixels in lookup table.")

            if len(pixel_list) > 0:
                # Convert list of (x, y) pixel coordinates to ROI for GMM
                roi = np.array([frame[y, x] for x, y in pixel_list])
                roi = np.float32(roi)

                # Initialize GMM model and set the number of clusters
                model = cv.ml.EM_create()
                model.setClustersNumber(4)

                # Train the GMM model

                model.trainEM(roi)
                print(f"Successfully trained GMM model for label {label}.")

                # Store the GMM model
                colors.append(model)


            else:
                print(f"Not enough pixels to train GMM model for label {label}, skipping.")
                colors.append(None)

            pixel_label_list.append(pixel_list)

        color_models.append(colors)
        pixel_label_list_cameras.append(pixel_label_list)

    return new_voxel_list, new_colors, color_models, pixel_label_list_cameras


def paint_image(image, pixel_list):
    # auxiliar function to check in a view
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


    """cv.imshow('Painted Image in offline', image_aux)
    cv.imshow('Image normal', image)

    cv.waitKey(0)"""


def get_colour(label):
    # Get color from a label
    if label == 0:
        return [0, 0, 255]
    elif label == 1:
        return [0, 255, 0]
    elif label == 2:
        return [255, 0, 0]
    elif label == 3:
        return [255, 255, 0]


def color_model(voxel_list, frames_cam, lookup_table_selected_camera, selected_camera, lookup_table_every_camera):
    # Remove outliers from the entire voxel_list before clustering
    voxels_filtered, _ = remove_outliers_iqr(np.array(voxel_list))
    # Clustering
    labels, centers = clustering(voxels_filtered)
    # Construct color model
    new_voxel_list, new_colors, color_models, pixel_label_list_cameras = construct_color_model(voxels_filtered, labels,
                                                                                               centers,
                                                                                               lookup_table_every_camera,
                                                                                               frames_cam)

    # Auxiliar code to check
    selected_camera_aux = 1
    paint_image(frames_cam[selected_camera_aux], pixel_label_list_cameras[selected_camera_aux])
    return new_voxel_list, new_colors, color_models


def online_phase(colors_model, voxel_list, frames_cam, lookup_table_every_camera, curr_time, debug=True):
    print("---Online phase--->", " Time: ", curr_time)
    voxels_filtered, _ = remove_outliers_iqr(np.array(voxel_list))

    labels, centers = clustering(voxels_filtered)
    voxels_filtered = np.float32(voxels_filtered)

    frames = read_all_frames()

    calculated_labes = [None, None, None, None]
    new_voxel_list = []
    final_voxel_list = []
    number_voxels_label = []
    new_colors = []
    labels = np.ravel(labels)
    
    # Iterate over cameras
    for i in range(4):
        print(f"Camera {i + 1}")
        probabilities_labels = []
        aux_pixel_list = []

        # Obtain index for this frame and get it according to current time
        index = int(curr_time/5)

        frame = cv.cvtColor(frames[i][index], cv.COLOR_BGR2HSV)

        for label in range(len(np.unique(labels))):
            print(f"Label {label}")
            # voxels that have this label
            label_voxel = []

            # print(voxels_filtered.shape, " -- ", len(labels))
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
                # pixel = lookup_table_selected_camera.get(tuple(voxel), None)
                pixel = lookup_table_every_camera[i + 1].get(tuple(voxel), None)

                if pixel:
                    pixel_list.append(pixel)
                    label_voxel.append(voxel)
                    # we only save the number voxels once (Same for everyone)
                    if i == 0:
                        new_voxel_list.append(voxel)
                        auxiliar_number_voxels_label += 1
                        # auxiliar
                        new_color = get_colour(label)
                        new_colors.append(new_color)

            aux_pixel_list.append(pixel_list)

            # Variable used to know how many voxels are in each label and change color
            if i == 0:
                number_voxels_label.append(auxiliar_number_voxels_label)

            # first part of online - match colour with cluster
            probabilities = []
            for j in range(len(colors_model[i])):
                roi = np.array([frame[y, x] for x, y in pixel_list])
                roi = np.float32(roi)
                total_prob = 0
                # probability for this color model
                for sample in roi:
                    (prob, _), _ = colors_model[i][j].predict2(sample)
                    total_prob += prob
                probabilities.append(total_prob)
            probabilities_labels.append(probabilities)

        # Linear sum assignmetn to match persons-colors
        probabilities_array = np.array(probabilities_labels)
        
        _, new_labels = linear_sum_assignment(probabilities_array)
        calculated_labes[i] = new_labels.tolist()


        # see decision of each camera:
        # We are using camera 1 as is the one with most accurate predictions according to our tests
        if i == 0:
            index = int(curr_time / 5)
            frame_aux = frames[i][index]
            labels_from_this_cam = calculated_labes[i]
            reordered_list = [aux_pixel_list[label] for label in labels_from_this_cam]
            final_colors = []
            cont = 0
            aux_cont = 0
      
            while cont < number_voxels_label[3]:
                new_decision_label = labels_from_this_cam[aux_cont]
                aux_color = get_colour(new_decision_label)
                final_colors.append(aux_color)
                if cont == number_voxels_label[aux_cont]:
                    aux_cont += 1
                cont += 1

            paint_image(frame_aux, reordered_list)

    if debug:
        # Auxiliar code to check probs.
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


    cluster_asssigned = 0
    # just trust camera 2
    final_labels = calculated_labes[1]
    final_colors = []
    # for label in labels:

    # Assuming 'centers' is a list of cluster centroids from the current frame
    # and 'final_labels' contains the index of the person each cluster is matched to
    for i, center in enumerate(centers):
        person_id = final_labels[i]  # Get the matched person ID for this cluster
        # Append the person ID, cluster center (x, y), and current frame number
        trajectory_data.append((person_id, center[0], center[1], curr_time))

    return new_voxel_list, new_colors



def plot_trajectories(trajectory_data):
    plt.figure(figsize=(10, 6))
    # Iterate over persons
    for person_id in set(data[0] for data in trajectory_data):
        # Extract x and y positions for this person across frames
        x_positions = [data[2] for data in trajectory_data if data[0] == person_id]
        y_positions = [data[3] for data in trajectory_data if data[0] == person_id]

        # Ensure y_positions are flipped if necessary to match the plotting coordinate system
        y_positions = [max(y_positions) - y for y in y_positions]

        plt.plot(x_positions, y_positions, marker='o', label=f'Person {person_id}')

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Trajectories of People Over Time')
    plt.legend()
    plt.show()
