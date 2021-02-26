import os
import argparse
import cv2 as cv
import numpy as np
import open3d as o3d
from torch import threshold
from torch.utils.data import DataLoader

from sensor_dataset import sensor_dataset

import torchvision.transforms.functional as TF

import copy

ap = argparse.ArgumentParser()

ap.add_argument('-l', '--input_lidar_file_path', type=str, required=True)
ap.add_argument('-i', '--input_img_file_path', type=str, required=True)
ap.add_argument('-p', '--input_pose_file_path', type=str, required=True)

args = vars(ap.parse_args())

batch_size = 1

dataset = sensor_dataset(lidar_dataset_path=args['input_lidar_file_path'], 
                        img_dataset_path=args['input_img_file_path'], 
                        pose_dataset_path=args['input_pose_file_path'],
                        train_sequence=['00', '01', '02'], valid_sequence=['01'], test_sequence=['02'],
                        h_fov=[-45, 45], h_res=0.2, v_fov=[-17.5, 24.9], v_res=0.4)

dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)

dataloader.dataset.mode = 'training'
for batch_idx, (lidar_range_img_tensor, x_in_range_img_tensor, y_in_range_img_tensor, pcd_dist_normalized_tensor, current_img_tensor, pose_6DOF_tensor) in enumerate(dataloader):
    
    lidar_img_list = []
    for batch_index in range(lidar_range_img_tensor.size(0)):

        lidar_range_img = np.array(TF.to_pil_image(lidar_range_img_tensor[batch_index]))
        lidar_range_img = cv.resize(lidar_range_img, dsize=(int(1280/batch_size), int(240 * 0.55/batch_size)), interpolation=cv.INTER_CUBIC)
        lidar_range_img = cv.applyColorMap(lidar_range_img, cv.COLORMAP_HSV)

        lidar_img_list.append(lidar_range_img)
    
    lidar_total_img = cv.hconcat(lidar_img_list)

    x_in_range_img = x_in_range_img_tensor.numpy()[0, :]
    y_in_range_img = y_in_range_img_tensor.numpy()[0, :]
    pcd_dist_normalized = pcd_dist_normalized_tensor.numpy()[0, :]

    img_list = []
    for batch_index in range(current_img_tensor.size(0)):
        current_img = np.array(TF.to_pil_image(current_img_tensor[batch_index]))
        current_img = cv.cvtColor(current_img, cv.COLOR_RGB2BGR)
        current_img = cv.resize(current_img, dsize=(int(1280/batch_size), int(240/batch_size)), interpolation=cv.INTER_CUBIC)

        img_list.append(current_img)

    img_total = cv.hconcat(img_list)

    # ratio = (max(x_in_range_img) / (1280/batch_size))
    # mask_width = int(ratio * (1280/batch_size))
    # mask_height = int(ratio * (240/batch_size))
    # lidar_mask = np.ones([mask_height + 1, mask_width + 1, 3])
    # # lidar_mask = cv.resize(lidar_mask, dsize=(mask_width + 1, mask_height + 1), interpolation=cv.INTER_CUBIC)
    
    # lidar_mask[y_in_range_img, x_in_range_img] = np.array([pcd_dist_normalized, 0 * pcd_dist_normalized, 0 * pcd_dist_normalized]).transpose(1, 0)
    
    # lidar_mask = cv.resize(lidar_mask, dsize=(int(1280/batch_size), int(240/batch_size)), interpolation=cv.INTER_CUBIC)
    
    # final_output = lidar_mask
    
    blended_output = copy.deepcopy(img_total)
    offset = int(240 * 0.45/batch_size)
    for x in range(0, lidar_total_img.shape[1]):
        for y in range(0, lidar_total_img.shape[0]):
            blended_output[y + offset, x] = lidar_total_img[y, x]

    block = 255 * np.ones([int(240 * 0.2/batch_size), int(1280/batch_size), 3], dtype=np.uint8)

    final_output = cv.vconcat([lidar_total_img, img_total, block, blended_output])

    cv.imshow('3D LiDAR Range Image', final_output)
    cv.waitKey(1)