import glm
import random
import numpy as np
from auxiliar import get_intrinsics, extract_frames, averaging_background_model, subtract_background
import cv2
import os

block_size = 1.0


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors


def set_voxel_positions(width, height, depth):
    """
    Generates random voxel locations
    Calculate proper voxel arrays.
    :param :width
    :param :height
    :param :depth
    :return 
    """
    camera_folders = ["cam1", "cam2", "cam3", "cam4"]
    mtxs, dists, rvecss, tvecss, substracted, video = [], [],[], [],[],[]


    for folder in camera_folders:
        video_path = os.path.join('data',folder,'video.avi')
        background_path = os.path.join('data',folder,'background.avi')
        mtx, dist, rvecs, tvecs = get_intrinsics(folder)
        mtxs.append(mtx)
        dists.append(dist)
        rvecss.append(rvecs)
        tvecss.append(tvecs)
        
        frames = extract_frames(video_path)
        frame = frames[0]
        processed_frame,_ ,_= subtract_background(frame, averaging_background_model(background_path))
        substracted.append(processed_frame)
        video.append(frame)
 
    data, colors = [], []
    #Look up table
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                #TODO
                voxel = True
                color = []
                i = 0
                for folder in camera_folders:
                    voxels = ((x - width / 2) * 50, (y - height / 2) * 50, -z * 50)
                    points,_ = cv2.projectPoints(voxels, rvecss[i], tvecss[i], mtxs[i], dists[i])
                    points = np.reshape(points[::-1], (2, -1)) 
                    subs = substracted[i]
                    heightMask, widthMask, _ = subs.shape
                    #print(heightMask, widthMask)
                    #print(points)
                    points[0] = abs(points[0])
                    points[1] = abs(points[1])

                    if 0 <= points[0] < heightMask and 0 <= points[1] < widthMask:
                        val = subs[int(points[0]), int(points[1])]
                        color.append(video[i][int(points[0])][int(points[1])])

                        # If the value is zero in any of the cameras it is not considered
                        # as activated in the 3D space.
                        if val.any() == 0:
                            voxel = False

                    if voxel:
                        data.append(
                            [(x * block_size/2 - width)+50, (z * block_size)/2,
                            (y * block_size/2 - depth)+50])
                        final_color = np.mean(np.array(color), axis=0) / 256
                        colors.append(final_color)
                   
                    i+= 1

    return data, colors



def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    camera_folders = ["cam1", "cam2", "cam3", "cam4"]

    cam_positions = []
    for camera_path in camera_folders:
        video_path = os.path.join(camera_path, 'video.avi')
        mtx, dist, rvecs, tvecs = get_intrinsics(camera_path)
        #print(rvecs, "--", tvecs)
        tvecs /= 10
        rotation_matrix = cv2.Rodrigues(rvecs)[0]
        position_vector = -np.matrix(rotation_matrix).T * np.matrix(tvecs)
        cam_positions.append([position_vector[0][0], -position_vector[2][0], -position_vector[1][0]])
    return cam_positions, [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


def get_cam_rotation_matrices():
    cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    camera_rotations = []
    camera_folders = ["cam1", "cam2", "cam3", "cam4"]

    for folder in camera_folders:
        mtx, dist, rvecs, tvecs = get_intrinsics(folder)

        rotation_matrix = cv2.Rodrigues(rvecs)[0]

        # Form a 4x4 transformation matrix including rotation and translation for OpenGL
        transformation_matrix = glm.mat4(rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2], tvecs[0][0],
                                         rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], tvecs[1][0],
                                         rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2], tvecs[2][0],
                                         0, 0, 0, 1)
        # Apply GLM rotations to align with OpenGL's coordinate system
        adjusted_matrix = glm.rotate(transformation_matrix, glm.radians(-90), (0, 1, 0))
        adjusted_matrix = glm.rotate(adjusted_matrix, glm.radians(180), (1, 0, 0))

        camera_rotations.append(adjusted_matrix)

    return camera_rotations


