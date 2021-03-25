#!/bin/sh

image_dataset_path="/media/luwis/Linux Workspace/ICSL_Project/Visual SLAM/KITTI_data_odometry_color/dataset/sequences"
pose_dataset_path="/media/luwis/Linux Workspace/ICSL_Project/Visual SLAM/KITTI_data_odometry_color/data_odometry_poses/dataset/poses"
pretrained_model_path="/"

# 'training'
# 'tsne'
mode='tsne' 

batch_size=1

cuda_num='0'

training_epoch=100

python3 Agglo_tSNE_visualization.py --mode "$mode" \
                                    --input_img_file_path "$image_dataset_path" \
                                    --input_pose_file_path "$pose_dataset_path" \
                                    --pretrained_model_path "$pretrained_model_path" \
                                    --input_batch_size "$batch_size" \
                                    --input_CUDA_num "$cuda_num" \
                                    --training_epoch "$training_epoch" \

exit 0