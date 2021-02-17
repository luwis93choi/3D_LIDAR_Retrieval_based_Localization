# Binary 3D point cloud to PCD : https://github.com/cuge1995/bin-to-pcd-python/blob/master/bin2pcd.py

import numpy as np
import struct
import sys
import argparse
import open3d as o3d

class bin_to_pcd():
        
    def convert(self, binFileName):
        size_float = 4
        list_pcd = []

        with open(binFileName, 'rb') as f:
            
            byte = f.read(size_float * 4)
            
            while byte:
                x, y, z, intensity = struct.unpack('ffff', byte)
                list_pcd.append([x, y, z])
                byte = f.read(size_float * 4)

            np_pcd = np.asarray(list_pcd)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np_pcd)

            return pcd
        
    def convert_and_save(self, binFileName, pcdFileName):
        size_float = 4
        list_pcd = []

        with open(binFileName, 'rb') as f:
            
            byte = f.read(size_float * 4)
            
            while byte:
                x, y, z, intensity = struct.unpack('ffff', byte)
                list_pcd.append([x, y, z])
                byte = f.read(size_float * 4)

            np_pcd = np.asarray(list_pcd)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np_pcd)

            o3d.io.write_point_cloud(pcdFileName, pcd)

            return pcd, pcdFileName