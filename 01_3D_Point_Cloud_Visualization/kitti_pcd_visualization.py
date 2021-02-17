import numpy as np
import struct
import sys
import os
import argparse
import open3d as o3d

from bin_to_pcd import bin_to_pcd

ap = argparse.ArgumentParser()

ap.add_argument('-i', '--input_binary_file_path', type=str, required=True)
ap.add_argument('-o', '--output_pcd_file_path', type=str, required=True)

args = vars(ap.parse_args())

lidar_files = sorted(os.listdir(args['input_binary_file_path'] + '/velodyne'))

pcd_converter = bin_to_pcd()
pcd = pcd_converter.convert(binFileName=args['input_binary_file_path'] + '/velodyne/' + lidar_files[0])

vis = o3d.visualization.Visualizer()
vis.create_window()

for data in lidar_files:

    pcd = pcd_converter.convert(binFileName=args['input_binary_file_path'] + '/velodyne/' + data)

    original_shape = np.asarray(pcd.points).shape

    # for i in range(288000 - original_shape[0]):

    #     pcd.points.append([0, 0, 0])

    #print('{} / Original shape : {} / Appended shape : {}'.format(data, original_shape, np.asarray(pcd.points).shape))

    print('{} / Original shape : {}'.format(data, original_shape))

    #print(np.asarray(pcd.points))

    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.remove_geometry(pcd)

vis.destroy_window()
