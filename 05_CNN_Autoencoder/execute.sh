#!/bin/sh

lidar_dataset_path="/media/luwis/Linux Workspace/ICSL_Project/Visual SLAM/KITTI_data_odometry_color/data_odometry_velodyne/dataset/sequences"
image_dataset_path="/media/luwis/Linux Workspace/ICSL_Project/Visual SLAM/KITTI_data_odometry_color/dataset/sequences"
pose_dataset_path="/media/luwis/Linux Workspace/ICSL_Project/Visual SLAM/KITTI_data_odometry_color/data_odometry_poses/dataset/poses"

batch_size=4

cuda_num='0'

training_epoch=100

python3 main.py --input_lidar_file_path "$lidar_dataset_path" \
                --input_img_file_path "$image_dataset_path" \
                --input_pose_file_path "$pose_dataset_path" \
                --input_batch_size "$batch_size" \
                --input_CUDA_num "$cuda_num" \
                --training_epoch "$training_epoch" \

exit 0