#!/bin/sh

lidar_dataset_path="/media/luwis/Linux Workspace/ICSL_Project/Visual SLAM/KITTI_data_odometry_color/data_odometry_velodyne/dataset/sequences"
image_dataset_path="/media/luwis/Linux Workspace/ICSL_Project/Visual SLAM/KITTI_data_odometry_color/dataset/sequences"
pose_dataset_path="/media/luwis/Linux Workspace/ICSL_Project/Visual SLAM/KITTI_data_odometry_color/dataset/poses"
trained_model="/home/byungchanchoi/ICSL_Project/3D_LIDAR_Retrieval_based_Localization/06_Encoded_Feature_tSNE_Visualization/1616274861.8618088/Autoencoder.pth"

batch_size=1

cuda_num='2'

training_epoch=100

save_plot_animation=1

python3 tSNE_visualizer.py --input_lidar_file_path "$lidar_dataset_path" \
                           --input_img_file_path "$image_dataset_path" \
                           --input_pose_file_path "$pose_dataset_path" \
                           --input_batch_size "$batch_size" \
                           --input_CUDA_num "$cuda_num" \
                           --training_epoch "$training_epoch" \
                           --trained_model "$trained_model" \
                           --save_plot_animation "$save_plot_animation" \

exit 0