#!/bin/sh

lidar_dataset_path="/home/byungchanchoi/KITTI_dataset/data_odometry_velodyne/dataset/sequences"
pose_dataset_path="/home/byungchanchoi/KITTI_dataset/data_odometry_poses/dataset/poses"
pretrained_model_path="/"

# 'training'
# 'tsne'
mode='training' 

batch_size=8

cuda_num='0'

training_epoch=100

python3 Agglo_tSNE_visualization.py --mode "$mode" \
                                    --input_lidar_file_path "$lidar_dataset_path" \
                                    --input_pose_file_path "$pose_dataset_path" \
                                    --pretrained_model_path "$pretrained_model_path" \
                                    --input_batch_size "$batch_size" \
                                    --input_CUDA_num "$cuda_num" \
                                    --training_epoch "$training_epoch" \

exit 0