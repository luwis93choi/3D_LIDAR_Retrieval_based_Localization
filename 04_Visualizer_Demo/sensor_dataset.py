import os
import os.path
import numpy as np
import csv

import torch
import torch.utils.data
import torchvision.transforms.functional as TF

import PIL
from PIL import Image

from lidar_range_img_generator import range_img_generator

class dataset_dict_generator():

    def __init__(self, lidar_dataset_path='', img_dataset_path='', pose_dataset_path='',
                       train_sequence=['00'], valid_sequence=['01'], test_sequence=['02']):

        self.lidar_dataset_path = lidar_dataset_path
        self.img_dataset_path = img_dataset_path
        self.pose_dataset_path = pose_dataset_path


        dataset_sequence_list = np.array([train_sequence, valid_sequence, test_sequence], dtype=np.object)
        
        self.train_dataset_dict_path = './Train_lidar_dataset.csv'
        self.valid_dataset_dict_path = './Valid_lidar_dataset.csv'
        self.test_dataset_dict_path = './Test_lidar_dataset.csv'

        self.train_len = 0
        self.valid_len = 0
        self.test_len = 0

        ######################################
        ### Dataset Dictionary Preparation ###
        ######################################
        dataset_dict_idx = 0
        for dataset_type in dataset_sequence_list:

            self.data_idx = 0

            if dataset_dict_idx == 0:
                self.dataset_dict = open(self.train_dataset_dict_path, 'w', encoding='utf-8', newline='')

            elif dataset_dict_idx == 1:
                self.dataset_dict = open(self.valid_dataset_dict_path, 'w', encoding='utf-8', newline='')

            elif dataset_dict_idx == 2:
                self.dataset_dict = open(self.test_dataset_dict_path, 'w', encoding='utf-8', newline='')

            self.dataset_writer = csv.writer(self.dataset_dict)

            header_list = ['current_index', 'Sequence_index', 'current_lidar_path', 'current_img_path', 'current_x [m]', 'current_y [m]', 'current_z [m]', 'current_roll [rad]', 'current_pitch [rad]', 'current_yaw [rad]']
            self.dataset_writer.writerow(header_list)

            for sequence_idx in np.array(dataset_type):

                lidar_base_path = self.lidar_dataset_path + '/' + sequence_idx + '/velodyne'
                lidar_data_name = sorted(os.listdir(lidar_base_path))
                
                img_base_path = self.img_dataset_path + '/' + sequence_idx + '/image_2'
                img_data_name = sorted(os.listdir(img_base_path))

                # Pose data accumulation
                lines = []
                pose_file = open(self.pose_dataset_path + '/' + sequence_idx + '.txt', 'r')
                while True:
                    line = pose_file.readline()
                    lines.append(line)
                    if not line: break
                pose_file.close()

                for lidar_name, img_name, line in zip(np.array(lidar_data_name), np.array(img_data_name), np.array(lines)):
                    
                    # Pose data re-organization into x, y, z, euler angles
                    pose_line = line
                    pose = pose_line.strip().split()
                    
                    current_pose_T = [float(pose[3]), float(pose[7]), float(pose[11])]
                    current_pose_Rmat = np.array([
                                                [float(pose[0]), float(pose[1]), float(pose[2])],
                                                [float(pose[4]), float(pose[5]), float(pose[6])],
                                                [float(pose[8]), float(pose[9]), float(pose[10])]
                                                ])

                    current_x = current_pose_T[0]
                    current_y = current_pose_T[1]
                    current_z = current_pose_T[2]

                    current_roll = np.arctan2(current_pose_Rmat[2][1], current_pose_Rmat[2][2])
                    current_pitch = np.arctan2(-1 * current_pose_Rmat[2][0], np.sqrt(current_pose_Rmat[2][1]**2 + current_pose_Rmat[2][2]**2))
                    current_yaw = np.arctan2(current_pose_Rmat[1][0], current_pose_Rmat[0][0])

                    data = [self.data_idx, sequence_idx, lidar_base_path + '/' + lidar_name, img_base_path + '/' + img_name, current_x, current_y, current_z, current_roll, current_pitch, current_yaw]

                    self.dataset_writer.writerow(data)

                    self.data_idx += 1

            if dataset_dict_idx == 0:
                self.train_len = self.data_idx

            elif dataset_dict_idx == 1:
                self.valid_len = self.data_idx

            elif dataset_dict_idx == 2:
                self.test_len = self.data_idx

            self.dataset_dict.close()

            dataset_dict_idx += 1

class sensor_dataset(torch.utils.data.Dataset):

    def __init__(self, lidar_dataset_path='', img_dataset_path='', pose_dataset_path='',
                       train_transform=None,
                       valid_transform=None,
                       test_transform=None,
                       train_sequence=['00'], valid_sequence=['01'], test_sequence=['02'],
                       mode='training', normalization=None,
                       h_fov=[-180, 180], h_res=0.2, v_fov=[-24.9, 2], v_res=0.4):

        self.lidar_dataset_path = lidar_dataset_path
        self.img_dataset_path = img_dataset_path
        self.pose_dataset_path = pose_dataset_path

        self.train_sequence = train_sequence
        self.train_transform = train_transform

        self.valid_sequence = valid_sequence
        self.valid_transform = valid_transform
        
        self.test_sequence = test_sequence
        self.test_transform = test_transform

        self.data_idx = 0

        self.len = 0

        self.mode = mode

        self.lidar_range_img_generator = range_img_generator(h_fov=h_fov, h_res=h_res, v_fov=v_fov, v_res=v_res)

        self.dataset_dict_generator = dataset_dict_generator(lidar_dataset_path=lidar_dataset_path, 
                                                             img_dataset_path=img_dataset_path, 
                                                             pose_dataset_path=pose_dataset_path,
                                                             train_sequence=train_sequence, valid_sequence=valid_sequence, test_sequence=test_sequence)

        self.train_data_list = []
        train_dataset_dict = open(self.dataset_dict_generator.train_dataset_dict_path, 'r', encoding='utf-8')
        train_reader = csv.reader(train_dataset_dict)
        next(train_reader)     # Skip header row
        for row_data in train_reader:
            self.train_data_list.append(row_data)
        train_dataset_dict.close()

        self.valid_data_list = []
        valid_dataset_dict = open(self.dataset_dict_generator.valid_dataset_dict_path, 'r', encoding='utf-8')
        valid_reader = csv.reader(valid_dataset_dict)
        next(valid_reader)     # Skip heaer row
        for row_data in valid_reader:
            self.valid_data_list.append(row_data)
        valid_dataset_dict.close()

        self.test_data_list = []
        test_dataset_dict = open(self.dataset_dict_generator.test_dataset_dict_path, 'r', encoding='utf-8')
        test_reader = csv.reader(test_dataset_dict)
        next(test_reader)      # Skip header row
        for row_data in test_reader:
            self.test_data_list.append(row_data)
        test_dataset_dict.close()

    def __getitem__(self, index):

        if self.mode == 'training':
            item = self.train_data_list[index]

        elif self.mode == 'validation':
            item = self.valid_data_list[index]

        elif self.mode == 'test':
            item = self.test_data_list[index]
        
        lidar_range_img = self.lidar_range_img_generator.convert_range_img(pcd_path=item[2], output_type='img_pixel')
        lidar_range_img = TF.to_tensor(lidar_range_img)

        current_img = Image.open(item[3])
        current_img = TF.to_tensor(current_img)

        pose_6DOF = [float(i) for i in item[4:]]
        pose_6DOF = torch.tensor(pose_6DOF)

        return lidar_range_img, current_img, pose_6DOF

    def __len__(self):

        if self.mode == 'training':
            return self.dataset_dict_generator.train_len

        elif self.mode == 'validation':
            return self.dataset_dict_generator.valid_len

        elif self.mode == 'test':
            return self.dataset_dict_generator.test_len

