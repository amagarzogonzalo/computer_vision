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
    mtxs, dists, rvecss, tvecss = [], [],[], [],[]

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
        processed_frame = subtract_background(frame, averaging_background_model(background_path))
        """cv2.imshow('Foreground', processed_frame)
        if cv2.waitKey(0):
            break
        cv2.destroyAllWindows()"""
 
    data, colors = [], []
    #Look up table
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                i = 0
                for folder in camera_folders:
                    voxels = ((x - width / 2) * 40, (y - height / 2) * 40, -z * 40)
                    points,_ = cv2.projectpoints(voxels, rvecss[i], tvecss[i], mtxs[i], dists[i])



                    if True:
                        data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
                        colors.append([x / width, z / depth, y / height])
                    i+= 1
    return data, colors


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    return [[-64 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, -64 * block_size],
            [-64 * block_size, 64 * block_size, -64 * block_size]], \
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations