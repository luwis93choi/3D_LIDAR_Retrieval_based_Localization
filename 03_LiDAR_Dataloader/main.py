import os
import argparse
import cv2 as cv
import numpy as np
import open3d as o3d
from torch import threshold
from torch.utils.data import DataLoader

from sensor_dataset import sensor_dataset

import torchvision.transforms.functional as TF

ap = argparse.ArgumentParser()

ap.add_argument('-l', '--input_lidar_file_path', type=str, required=True)
ap.add_argument('-i', '--input_img_file_path', type=str, required=True)
ap.add_argument('-p', '--input_pose_file_path', type=str, required=True)

args = vars(ap.parse_args())

batch_size = 1

dataset = sensor_dataset(lidar_dataset_path=args['input_lidar_file_path'], 
                        img_dataset_path=args['input_img_file_path'], 
                        pose_dataset_path=args['input_pose_file_path'],
                        train_sequence=['00', '01', '02'], valid_sequence=['01'], test_sequence=['02'])

dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)

dataloader.dataset.mode = 'training'
for batch_idx, (lidar_range_img_tensor, current_img_tensor, pose_6DOF_tensor) in enumerate(dataloader):

    lidar_img_list = []
    for batch_index in range(lidar_range_img_tensor.size(0)):

        lidar_range_img = np.array(TF.to_pil_image(lidar_range_img_tensor[batch_index]))
        lidar_range_img = cv.resize(lidar_range_img, dsize=(int(1280/batch_size), int(240/batch_size)), interpolation=cv.INTER_CUBIC)
        lidar_range_img = cv.applyColorMap(lidar_range_img, cv.COLORMAP_HSV)

        lidar_img_list.append(lidar_range_img)
    
    lidar_total_img = cv.hconcat(lidar_img_list)

    img_list = []
    for batch_index in range(current_img_tensor.size(0)):
        current_img = np.array(TF.to_pil_image(current_img_tensor[batch_index]))
        current_img = cv.cvtColor(current_img, cv.COLOR_RGB2BGR)
        current_img = cv.resize(current_img, dsize=(int(1280/batch_size), int(240/batch_size)), interpolation=cv.INTER_CUBIC)

        img_list.append(current_img)

    img_total = cv.hconcat(img_list)

    final_output = cv.vconcat([lidar_total_img, img_total])

    cv.imshow('3D LiDAR Range Image', final_output)
    cv.waitKey(1)