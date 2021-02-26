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

    def __init__(self, lidar_dataset_path='', img_dataset_path='', label_dataset_path='', transformation_dataset_path='',
                       train_sequence=['00'], valid_sequence=['01'], test_sequence=['02']):

        self.lidar_dataset_path = lidar_dataset_path
        self.img_dataset_path = img_dataset_path
        self.label_dataset_path = label_dataset_path
        self.transformation_dataset_path = transformation_dataset_path

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

            header_list = ['current_index', 'Sequence_index', 'current_lidar_path', 'current_img_path', 'label_path', 'tranformation_path']

            self.dataset_writer.writerow(header_list)

            for sequence_idx in np.array(dataset_type):

                lidar_base_path = self.lidar_dataset_path + '/' + sequence_idx + '/velodyne'
                lidar_data_name = sorted(os.listdir(lidar_base_path))
                
                img_base_path = self.img_dataset_path + '/' + sequence_idx + '/image_2'
                img_data_name = sorted(os.listdir(img_base_path))

                if dataset_type == 'training':
                    label_base_path = self.label_dataset_path + '/' + sequence_idx + '/label_2'
                    label_data_name = sorted(os.listdir(label_base_path))

                transformation_base_path = self.transformation_dataset_path + '/' + sequence_idx + '/calib'
                transformation_data_name = sorted(os.listdir(transformation_base_path))

                for lidar_name, img_name, label_name, transform_name in zip(np.array(lidar_data_name), np.array(img_data_name), np.array(label_data_name), np.array(transformation_data_name)):
                    
                    data = [self.data_idx, sequence_idx, lidar_base_path + '/' + lidar_name, \
                                                         img_base_path + '/' + img_name, \
                                                         label_base_path + '/' + label_name, \
                                                         transformation_base_path + '/' + transform_name]

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

    def __init__(self, lidar_dataset_path='', img_dataset_path='', label_dataset_path='', transformation_dataset_path='',
                       train_transform=None,
                       valid_transform=None,
                       test_transform=None,
                       train_sequence=['00'], valid_sequence=['01'], test_sequence=['02'],
                       mode='training', normalization=None,
                       h_fov=[-180, 180], h_res=0.2, v_fov=[-24.9, 2], v_res=0.4):

        self.lidar_dataset_path = lidar_dataset_path
        self.img_dataset_path = img_dataset_path
        self.label_dataset_path = label_dataset_path
        self.transformation_dataset_path = transformation_dataset_path

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
                                                             label_dataset_path=label_dataset_path, 
                                                             transformation_dataset_path=transformation_dataset_path,
                                                             train_sequence=train_sequence, valid_sequence=valid_sequence, test_sequence=test_sequence)

        self.train_data_list = []
        train_dataset_dict = open(self.dataset_dict_generator.train_dataset_dict_path, 'r', encoding='utf-8')
        train_reader = csv.reader(train_dataset_dict)
        next(train_reader)     # Skip header row
        for row_data in train_reader:
            self.train_data_list.append(row_data)
        train_dataset_dict.close()
        print('train dataset ready')

        self.valid_data_list = []
        valid_dataset_dict = open(self.dataset_dict_generator.valid_dataset_dict_path, 'r', encoding='utf-8')
        valid_reader = csv.reader(valid_dataset_dict)
        next(valid_reader)     # Skip heaer row
        for row_data in valid_reader:
            self.valid_data_list.append(row_data)
        valid_dataset_dict.close()
        print('validation dataset ready')

        self.test_data_list = []
        test_dataset_dict = open(self.dataset_dict_generator.test_dataset_dict_path, 'r', encoding='utf-8')
        test_reader = csv.reader(test_dataset_dict)
        next(test_reader)      # Skip header row
        for row_data in test_reader:
            self.test_data_list.append(row_data)
        test_dataset_dict.close()
        print('test dataset ready')

    def __getitem__(self, index):

        if self.mode == 'training':
            item = self.train_data_list[index]

        elif self.mode == 'validation':
            item = self.valid_data_list[index]

        elif self.mode == 'test':
            item = self.test_data_list[index]
        
        lidar_range_img, x_in_range_img, y_in_range_img, pcd_dist_normalized = self.lidar_range_img_generator.convert_range_img(pcd_path=item[2], output_type='img_pixel')
        lidar_range_img = TF.to_tensor(lidar_range_img)

        x_in_range_img = torch.tensor(x_in_range_img)
        y_in_range_img = torch.tensor(y_in_range_img)
        pcd_dist_normalized = torch.tensor(pcd_dist_normalized)

        current_img = Image.open(item[3])
        current_img = TF.to_tensor(current_img)

        label_list = []
        bbox_list = []
        label_file = open(item[4], 'r')
        while True:
            
            line = label_file.readline()
            if not line: break

            data = line.strip().split()
            label_list.append(data[0])
            bbox_list.append([float(data[4]), float(data[5]), float(data[6]), float(data[7])])
        
        label_file.close()
        
        return lidar_range_img, x_in_range_img, y_in_range_img, pcd_dist_normalized, current_img, label_list, bbox_list

    def __len__(self):

        if self.mode == 'training':
            return self.dataset_dict_generator.train_len

        elif self.mode == 'validation':
            return self.dataset_dict_generator.valid_len

        elif self.mode == 'test':
            return self.dataset_dict_generator.test_len

