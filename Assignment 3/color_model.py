import cv2 as cv
import numpy as np
from sklearn.mixture import GaussianMixture
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
from scipy.interpolate import interp1d


def clustering(voxel_list, N=4):
    print(voxel_list)
    voxel_list = np.array(voxel_list).astype(np.float32)[:, [0, 2]]
    print(voxel_list)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    ret, labels, centers = cv.kmeans(voxel_list, N, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    unique, counts = np.unique(labels, return_counts=True)

    with open('labels.txt', 'w') as f:
        f.write("Frequency of each label:\n")
        f.write(f"Label {unique}: {counts}\n\n")
        for label in labels:
            f.write(f"{label}\n")
    return labels, centers


def construct_color_model(voxel_list, labels, centers, selected_frame, lookup_table_selected_camera, selected_camera):
    labels = np.ravel(labels)
    frame = cv.cvtColor(selected_frame, cv.COLOR_BGR2HSV)

    color_models = []
    pixel_label_list = []

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
            pixel = lookup_table_selected_camera.get(tuple(voxel), None)
            if pixel:
                pixel_list.append(pixel)

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
            color_models.append(model)
        else:
            print(f"Not enough pixels to train GMM model for label {label}, skipping.")
            color_models.append(None)

        pixel_label_list.append(pixel_list)

    return color_models, pixel_label_list


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



def color_model(voxel_list, frames_cam, lookup_table_selected_camera, selected_camera):

    labels, centers = clustering(voxel_list)

    _, pixel_list  = construct_color_model(voxel_list, labels, centers, frames_cam[selected_camera], lookup_table_selected_camera, selected_camera)
    paint_image(frames_cam[selected_camera], pixel_list)

