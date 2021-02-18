import os
import argparse
import cv2 as cv
import numpy as np
import open3d as o3d
from torch import threshold

from sensor_dataloader import sensor_dataloader

import torchvision.transforms.functional as TF

ap = argparse.ArgumentParser()

ap.add_argument('-l', '--input_lidar_file_path', type=str, required=True)
ap.add_argument('-i', '--input_img_file_path', type=str, required=True)
ap.add_argument('-p', '--input_pose_file_path', type=str, required=True)

args = vars(ap.parse_args())

dataloader = sensor_dataloader(lidar_dataset_path=args['input_lidar_file_path'], 
                                img_dataset_path=args['input_img_file_path'], 
                                pose_dataset_path=args['input_pose_file_path'],
                                train_sequence=['00', '01', '02'], valid_sequence=['01'], test_sequence=['02'])

dataloader.mode = 'test'
for batch_idx, (lidar_range_img_tensor, current_img_tensor, pose_6DOF_tensor) in enumerate(dataloader):

    lidar_range_img = np.array(TF.to_pil_image(lidar_range_img_tensor))
    lidar_range_img = cv.resize(lidar_range_img, dsize=(1280, 240), interpolation=cv.INTER_CUBIC)
    lidar_range_img = cv.applyColorMap(lidar_range_img, cv.COLORMAP_HSV)

    current_img = np.array(TF.to_pil_image(current_img_tensor))
    current_img = cv.cvtColor(current_img, cv.COLOR_RGB2BGR)
    current_img = cv.resize(current_img, dsize=(1280, 240), interpolation=cv.INTER_CUBIC)

    img_total = cv.vconcat([lidar_range_img, current_img])

    cv.imshow('3D LiDAR Range Image', img_total)
    cv.waitKey(1)
