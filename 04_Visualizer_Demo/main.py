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
ap.add_argument('-b', '--input_label_file_path', type=str, required=True)
ap.add_argument('-t', '--input_transformation_file_path', type=str, required=True)

args = vars(ap.parse_args())

batch_size = 1

h_fov = [-45, 45]
h_res = 0.2
v_fov=[-17.5, 17.5]
v_res=0.4

dataset = sensor_dataset(lidar_dataset_path=args['input_lidar_file_path'], 
                        img_dataset_path=args['input_img_file_path'], 
                        label_dataset_path=args['input_label_file_path'], 
                        transformation_dataset_path=args['input_transformation_file_path'],
                        train_sequence=['training'], valid_sequence=['testing'], test_sequence=['testing'],
                        h_fov=h_fov, h_res=h_res, v_fov=v_fov, v_res=v_res)

dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

dataloader.dataset.mode = 'training'
for batch_idx, (lidar_range_img_tensor, x_in_range_img_tensor, y_in_range_img_tensor, pcd_dist_normalized_tensor, \
                current_img_tensor, \
                label_list, bbox_list) in enumerate(dataloader):
    
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
    left_coord_ratio = []
    top_coord_ratio = []
    right_coord_ratio = []
    bottom_coord_ratio = []
    for batch_index in range(current_img_tensor.size(0)):
        current_img = np.array(TF.to_pil_image(current_img_tensor[batch_index]))
        current_img = cv.cvtColor(current_img, cv.COLOR_RGB2BGR)

        for i in range(len(bbox_list)):

            # Saving original height, width ratio of bounding box within original image
            left_coord_ratio.append((bbox_list[i][0] / current_img.shape[1]).item())
            top_coord_ratio.append((bbox_list[i][1] / current_img.shape[0]).item())
            right_coord_ratio.append((bbox_list[i][2] / current_img.shape[1]).item())
            bottom_coord_ratio.append((bbox_list[i][3] / current_img.shape[0]).item())

            cv.rectangle(current_img, (int(bbox_list[i][0]), int(bbox_list[i][1])), (int(bbox_list[i][2]), int(bbox_list[i][3])), (0, 255, 0), 3)

        current_img = cv.resize(current_img, dsize=(int(1280/batch_size), int(240/batch_size)), interpolation=cv.INTER_CUBIC)

        img_list.append(current_img)

    img_total = cv.hconcat(img_list)


    blended_output = copy.deepcopy(img_total)
    offset = int(240 * 0.45/batch_size)
    for x in range(0, lidar_total_img.shape[1]):
        for y in range(0, lidar_total_img.shape[0]):
            blended_output[y + offset, x] = lidar_total_img[y, x]
    
    # Use original height, width ratio within original image resolution in order to map the bounding box onto blended output
    for i in range(len(bbox_list)):
        cv.rectangle(blended_output, (int(blended_output.shape[1] * left_coord_ratio[i]), int(blended_output.shape[0] * top_coord_ratio[i])), \
                                     (int(blended_output.shape[1] * right_coord_ratio[i]), int(blended_output.shape[0] * bottom_coord_ratio[i])), (0, 0, 0), 3)
    
    Text1 = 255 * np.ones([int(240 * 0.2/batch_size), int(1280/batch_size), 3], dtype=np.uint8)
    cv.putText(Text1, '3D LiDAR Range Image | Horizontal FOV : {} ~ {} | Vertical FOV : {} ~ {}'.format(h_fov[0], h_fov[1], v_fov[0], v_fov[1]), \
              (10, int(240 * 0.2 * 0.6/batch_size)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0, 255), 2)

    Text2 = 255 * np.ones([int(240 * 0.2/batch_size), int(1280/batch_size), 3], dtype=np.uint8)
    cv.putText(Text2, 'Front Camera Image | Horizontal FOV : {} ~ {} | Vertical FOV : {} ~ {}'.format(h_fov[0], h_fov[1], v_fov[0], v_fov[1]), \
              (10, int(240 * 0.2 * 0.6/batch_size)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0, 255), 2)

    Text3 = 255 * np.ones([int(240 * 0.2/batch_size), int(1280/batch_size), 3], dtype=np.uint8)
    cv.putText(Text3, 'Blended Output (3D LiDAR Range Image + Camera Image + Bounding Box)', (10, int(240 * 0.2 * 0.6/batch_size)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0, 255), 2)

    final_output = cv.vconcat([Text1, lidar_total_img, Text2, img_total, Text3, blended_output])

    cv.imshow('3D LiDAR Range Image', final_output)
    cv.waitKey(1)