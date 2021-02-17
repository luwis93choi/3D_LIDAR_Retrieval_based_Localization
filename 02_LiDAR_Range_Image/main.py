from lidar_range_img_generator import range_img_generator

import os
import argparse
import cv2 as cv
import open3d as o3d

ap = argparse.ArgumentParser()

ap.add_argument('-l', '--input_binary_pcd_file_path', type=str, required=True)
ap.add_argument('-i', '--input_ref_img_file_path', type=str, required=True)

args = vars(ap.parse_args())

lidar_files = sorted(os.listdir(args['input_binary_pcd_file_path'] + '/velodyne'))
img_files = sorted(os.listdir(args['input_ref_img_file_path'] + '/image_2'))

img_generator = range_img_generator(h_fov=[-180, 180], h_res=0.2, v_fov=[-24.9, 2], v_res=0.4)

vis = o3d.visualization.Visualizer()
vis.create_window()

for lidar_data, ref_img_data in zip(lidar_files, img_files):

    range_img = img_generator.convert_range_img(pcd_path=args['input_binary_pcd_file_path'] + '/velodyne/' + lidar_data, output_type='img_pixel')
    resized_range_img = cv.resize(range_img, dsize=(640, 120), interpolation=cv.INTER_CUBIC)
    resized_range_img = cv.applyColorMap(resized_range_img, cv.COLORMAP_JET)

    pcd = img_generator.convert_bin_to_pcd(bin_path=args['input_binary_pcd_file_path'] + '/velodyne/' + lidar_data)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.remove_geometry(pcd)

    ref_img = cv.imread(args['input_ref_img_file_path'] + '/image_2/' + ref_img_data)
    resized_ref_img = cv.resize(ref_img, dsize=(640, 120), interpolation=cv.INTER_CUBIC)

    img_total = cv.vconcat([resized_range_img, resized_ref_img])

    cv.imshow('3D LiDAR Range Image', img_total)
    cv.waitKey(1)