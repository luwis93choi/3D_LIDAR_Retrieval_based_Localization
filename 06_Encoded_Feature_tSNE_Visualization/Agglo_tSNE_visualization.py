from agg_cluster_dataset import dataset_dict_generator

dataset = dataset_dict_generator(img_dataset_path='/media/luwis/Linux Workspace/ICSL_Project/Visual SLAM/KITTI_data_odometry_color/dataset/sequences', 
                                 pose_dataset_path='/media/luwis/Linux Workspace/ICSL_Project/Visual SLAM/KITTI_data_odometry_color/data_odometry_poses/dataset/poses',
                                 sequence_to_use=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', ], 
                                 train_ratio=0.3, valid_ratio=0.3, test_ratio=0.3,
                                 cluster_linkage='ward', cluster_distance=2.0)

