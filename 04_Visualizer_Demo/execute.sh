#!/bin/sh

lidar_dataset_path="/media/luwis/Linux Workspace/ICSL_Project/LiDAR Object Detection/data_object_velodyne"
image_dataset_path="/media/luwis/Linux Workspace/ICSL_Project/LiDAR Object Detection/data_object_image_2"
label_dataset_path="/media/luwis/Linux Workspace/ICSL_Project/LiDAR Object Detection/data_object_label_2"
transformation_dataset_path="/media/luwis/Linux Workspace/ICSL_Project/LiDAR Object Detection/data_object_calib"

python3 main.py --input_lidar_file_path "$lidar_dataset_path" \
                --input_img_file_path "$image_dataset_path" \
                --input_label_file_path "$label_dataset_path" \
                --input_transformation_file_path "$transformation_dataset_path"

exit 0