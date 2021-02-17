# Reference 01 : KITTI Tutorial (https://github.com/windowsub0406/KITTI_Tutorial)
# Reference 02 : Vehicle Detection from 3D Lidar Using Fully Convolutional Network (http://www.roboticsproceedings.org/rss12/p42.pdf)

import numpy as np
import struct
import sys
import argparse
import open3d as o3d

class range_img_generator():

    def __init__(self, h_fov, h_res, v_fov, v_res):

        self.h_fov = h_fov
        self.h_res = h_res

        self.v_fov = v_fov
        self.v_res = v_res

        self.pcd = None

    def convert_bin_to_pcd(self, bin_path=None):

        size_float = 4
        list_pcd = []

        with open(bin_path, 'rb') as f:

            byte = f.read(size_float * 4)   # Read the bytes for 4 float values (x, y, z, intensity)

            while byte:
                x, y, z, intensity = struct.unpack('ffff', byte)    # Convert the bytes into actual float values
                list_pcd.append([x, y, z])

            np_pcd = np.asarray(list_pcd)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np_pcd)

        self.pcd = pcd

    def convert_range_img(self, pcd_path=None):

        if pcd_path == None:

            print('[Error] No Data Path')

            return None

        else:

            