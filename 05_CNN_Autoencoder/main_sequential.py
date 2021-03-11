import os
import argparse
import cv2 as cv
import numpy as np
import open3d as o3d
from torch import threshold
from torch.utils.data import DataLoader

from sequential_sensor_dataset import sequential_sensor_dataset

import torchvision.transforms.functional as TF

ap = argparse.ArgumentParser()

ap.add_argument('-l', '--input_lidar_file_path', type=str, required=True)
ap.add_argument('-i', '--input_img_file_path', type=str, required=True)
ap.add_argument('-p', '--input_pose_file_path', type=str, required=True)

args = vars(ap.parse_args())

batch_size = 1

dataset = sequential_sensor_dataset(lidar_dataset_path=args['input_lidar_file_path'], 
                                    img_dataset_path=args['input_img_file_path'], 
                                    pose_dataset_path=args['input_pose_file_path'],
                                    train_sequence=['00', '01', '02'], valid_sequence=['01'], test_sequence=['02'],
                                    sequence_length=10)

dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

dataloader.dataset.mode = 'test'
for batch_idx, (lidar_range_img_stack_tensor, current_img_tensor, pose_6DOF_tensor) in enumerate(dataloader):

    # Data Dimension Standard : Batch Size x Sequence Length x Data Shape
    # Sequential LiDAR = Batch Size x Sequence Length x 68 (Height) x 1800 (Width)
    # Sequential Image = Batch Size x Sequence Length x 3 (Channel) x 376 (Height) x 1241 (Width)
    # Sequential Pose = Batch Size x Sequence Length x 6 (6 DOF)

    print(lidar_range_img_stack_tensor.size())
    print(current_img_tensor.size())
    print(pose_6DOF_tensor.size())
    print('---------------------------------')

    ### Sequential LiDAR Stack Display ###
    lidar_img_sequence_list = []
    lidar_total_img = []
    seq_len = dataloader.dataset.sequence_length
    for batch_index in range(lidar_range_img_stack_tensor.size(0)):

        for seq in range(dataloader.dataset.sequence_length):
            lidar_range_img = np.array(TF.to_pil_image(lidar_range_img_stack_tensor[batch_index][seq]))
            lidar_range_img = cv.resize(lidar_range_img, dsize=(int(1280/seq_len), int(240/(seq_len * 0.5))), interpolation=cv.INTER_CUBIC)
            lidar_range_img = cv.applyColorMap(lidar_range_img, cv.COLORMAP_HSV)

            lidar_img_sequence_list.append(lidar_range_img)

        lidar_total_img.append(cv.hconcat(lidar_img_sequence_list))
        lidar_img_sequence_list = []
    
    final_lidar_img_output = cv.vconcat(lidar_total_img)

    ### Sequential Image Stack Display ###
    img_sequence_list = []
    total_img = []
    seq_len = dataloader.dataset.sequence_length
    for batch_index in range(current_img_tensor.size(0)):

        for seq in range(dataloader.dataset.sequence_length):
            current_img = np.array(TF.to_pil_image(current_img_tensor[batch_index][seq]))
            current_img = cv.cvtColor(current_img, cv.COLOR_RGB2BGR)    # Re-Order the image array into BGR for display purpose
            current_img = cv.resize(current_img, dsize=(int(1280/seq_len), int(240/(seq_len * 0.5))), interpolation=cv.INTER_CUBIC)

            img_sequence_list.append(current_img)

        total_img.append(cv.hconcat(img_sequence_list))
        img_sequence_list = []
    
    final_img_output = cv.vconcat(total_img)

    final_output = cv.vconcat([final_lidar_img_output, final_img_output])

    cv.imshow('3D LiDAR Range Image', final_output)
    cv.waitKey(1)