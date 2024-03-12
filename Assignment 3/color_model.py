import cv2 as cv
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
from scipy.interpolate import interp1d


def remove_outliers_from_clusters(voxel_list, labels, centers):

    distances = np.linalg.norm(voxel_list - centers[labels], axis=1)

    threshold = np.percentile(distances, 95)

    filter_mask = distances < threshold
    voxels_filtered = voxel_list[filter_mask]

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    ret, labels_filtered, centers_filtered = cv.kmeans(voxels_filtered.astype(np.float32), len(centers), None, criteria, 10, cv.KMEANS_PP_CENTERS)

    return voxels_filtered, labels_filtered, centers_filtered



def clustering(voxel_list, N=4):
    voxel_list = np.array(voxel_list).astype(np.float32)[:, [0, 2]]

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    ret, labels, centers = cv.kmeans(voxel_list, N, None, criteria, 10, cv.KMEANS_PP_CENTERS)

    #ret, labels, centers = cv.kmeans(voxel_list, N, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    unique, counts = np.unique(labels, return_counts=True)

    with open('labels.txt', 'w') as f:
        f.write("Frequency of each label:\n")
        f.write(f"Label {unique}: {counts}\n\n")
        for label in labels:
            f.write(f"{label}\n")


    return labels, centers


def construct_color_model(voxel_list, labels, centers, lookup_table_every_camera, frames_cam):
    color_models = []
    pixel_label_list_cameras = []

    for i in range(4):
        labels = np.ravel(labels)
        frame = cv.cvtColor(frames_cam[i], cv.COLOR_BGR2HSV)
        cv.imshow("auxiliar", frames_cam[i])
        cv.waitKey(0)

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
            
                    new_color = [0,0,255]
                    #new_color = np.array([0,0,255], dtype=np.float32)
                    
                    if label == 1:
                        new_color = [0,255,0]
                    elif label == 2:
                        new_color = [255,0,0]
                    elif label == 3:
                        new_color = [255,255,0]
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
    cv.imshow('Painted Image', image_aux)
    cv.imshow('Image normal', image)

    cv.waitKey(0)
    cv.destroyAllWindows()


def online(voxel_list, labels, centers, lookup_table_every_camera, frames_cam):
    color_models = []

    for i in range(4):
        labels = np.ravel(labels)
        frame = cv.cvtColor(frames_cam[i], cv.COLOR_BGR2HSV)

        for label in range(len(np.unique(labels))):

            voxels_person = np.array(voxel_list)[labels == label]

            # Calculate the 't-shirt' and 'head' cutoffs
            tshirt = np.mean(voxels_person[:, 1])
            head = np.max(voxels_person[:, 1])
            # Create ROI based on 'tshirt' and 'head' values
            voxel_roi = (voxels_person[:, 1] > tshirt) & (voxels_person[:, 1] < 3 / 4 * head)
            voxels_person_roi = voxels_person[voxel_roi]

            pixel_list = []
            for voxel in voxels_person_roi:
                pixel = lookup_table_every_camera[i + 1].get(tuple(voxel), None)
                if pixel:
                    pixel_list.append(pixel)

            if len(pixel_list) > 0:
                # Convert list of (x, y) pixel coordinates to ROI for GMM
                roi = np.array([frame[y, x] for x, y in pixel_list])
                roi = np.float32(roi)

                # Create and train the GMM (EM) model
                model = GaussianMixture(n_components=4, covariance_type='full')
                model.fit(roi[:, :2])

                color_models.append(model)
            else:
                color_models.append(None)

    return color_models

def calculate_gmm_distance(gmm1, gmm2):
    distance = np.sum(np.linalg.norm(gmm1.means_ - gmm2.means_, axis=1))
    return distance


def match_and_track(current_color_models, offline_color_models):
    num_current = len(current_color_models)
    num_offline = len(offline_color_models)
    cost_matrix = np.zeros((num_current, num_offline))

    for i, current_model in enumerate(current_color_models):
        for j, offline_model in enumerate(offline_color_models):
            cost_matrix[i, j] = calculate_gmm_distance(current_model, offline_model)

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Map current clusters to individuals based on matching
    mapping = dict(zip(row_indices, col_indices))
    return mapping


def process_frame_and_track(voxel_list, frames_cam, lookup_table_every_camera, offline_color_models):

    labels, centers = clustering(voxel_list)

    # Construct color models for the current frame
    _, _, current_color_models, _ = construct_color_model(voxel_list, labels, centers, lookup_table_every_camera,frames_cam)

    # Extract GMMs for comparison
    current_gmms = []
    for model in current_color_models:
        if model is not None:
            current_gmms.append(model)

    # Match current clusters to offline individuals and track them
    mapping = match_and_track(current_gmms, offline_color_models)

    # Track the 2D position of each individual based on cluster centers
    tracked_positions = {person_id: centers[cluster_id] for cluster_id, person_id in mapping.items()}

    return tracked_positions


def color_model(voxel_list, frames_cam, lookup_table_selected_camera, selected_camera, lookup_table_every_camera):

    labels, centers,  = clustering(voxel_list)

    new_voxel_list, new_colors, color_models, pixel_label_list_cameras  = construct_color_model(voxel_list, labels, centers, lookup_table_every_camera, frames_cam)
    selected_camera_aux = 1 # camera that we want to see in the view
    paint_image(frames_cam[selected_camera_aux], pixel_label_list_cameras[selected_camera_aux])
    return new_voxel_list, new_colors
