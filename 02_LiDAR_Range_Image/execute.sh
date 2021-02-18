#!/bin/sh

lidar_dataset_path="/media/luwis/Linux Workspace/ICSL_Project/Visual SLAM/KITTI_data_odometry_color/data_odometry_velodyne/dataset/sequences/01"
image_dataset_path="/media/luwis/Linux Workspace/ICSL_Project/Visual SLAM/KITTI_data_odometry_color/dataset/sequences/01"

python3 main.py --input_binary_pcd_file_path "$lidar_dataset_path" \
                --input_ref_img_file_path "$image_dataset_path" \

exit 0