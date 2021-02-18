import os
import argparse
import cv2 as cv
import open3d as o3d

from sensor_dataloader import sensor_dataloader

ap = argparse.ArgumentParser()

ap.add_argument('-l', '--input_lidar_file_path', type=str, required=True)
ap.add_argument('-i', '--input_img_file_path', type=str, required=True)
ap.add_argument('-p', '--input_pose_file_path', type=str, required=True)

args = vars(ap.parse_args())

dataset_gen = sensor_dataloader(lidar_dataset_path=args['input_lidar_file_path'], 
                                img_dataset_path=args['input_img_file_path'], 
                                pose_dataset_path=args['input_pose_file_path'],
                                train_sequence=['00', '01', '02'], valid_sequence=['01'], test_sequence=['02'])
