import os
import os.path
import numpy as np
import csv

import torch
import torch.utils.data
from torchvision import transforms
import torchvision.transforms.functional as TF

import PIL
from PIL import Image

import cv2 as cv

import random
import copy

from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from torchvision.transforms.transforms import RandomErasing, ToTensor

class dataset_dict_generator():

    def __init__(self, img_dataset_path='', pose_dataset_path='',
                       sequence_to_use=['00'], train_ratio=0, valid_ratio=0, test_ratio=0,
                       cluster_linkage='ward', cluster_distance=1.0):

        self.img_dataset_path = img_dataset_path
        self.pose_dataset_path = pose_dataset_path
        
        self.full_dataset_dict_path = './[Cluster]Full_lidar_dataset.csv'

        self.train_dataset_dict_path = './[Cluster]Train_lidar_dataset.csv'
        self.valid_dataset_dict_path = './[Cluster]Valid_lidar_dataset.csv'
        self.test_dataset_dict_path = './[Cluster]Test_lidar_dataset.csv'

        self.train_len = 0
        self.valid_len = 0
        self.test_len = 0

        self.agglo_clusterer = AgglomerativeClustering(n_clusters=None, linkage=cluster_linkage, distance_threshold=cluster_distance)

        self.seq_path_dict = {}

        ######################################
        ### Dataset Dictionary Preparation ###
        ######################################
        
        self.full_dataset_dict = open(self.full_dataset_dict_path, 'w', encoding='utf-8', newline='')
        self.train_dataset_dict = open(self.train_dataset_dict_path, 'w', encoding='utf-8', newline='')
        self.valid_dataset_dict = open(self.valid_dataset_dict_path, 'w', encoding='utf-8', newline='')
        self.test_dataset_dict = open(self.test_dataset_dict_path, 'w', encoding='utf-8', newline='')

        self.full_dataset_writer = csv.writer(self.full_dataset_dict)
        self.train_dataset_writer = csv.writer(self.train_dataset_dict)
        self.valid_dataset_writer = csv.writer(self.valid_dataset_dict)
        self.test_dataset_writer = csv.writer(self.test_dataset_dict)

        header_list = ['current_index', 'Sequence_num', 'Sequence_idx', 'current_img_path', 'current_x [m]', 'current_y [m]', 'current_z [m]', 'current_roll [rad]', 'current_pitch [rad]', 'current_yaw [rad]', 'cluster_label', 'type']
        self.full_dataset_writer.writerow(header_list)
        self.train_dataset_writer.writerow(header_list)
        self.valid_dataset_writer.writerow(header_list)
        self.test_dataset_writer.writerow(header_list)

        ### Iteration over all dataset sequences ###
        self.data_idx = 0
        for sequence_num in np.array(sequence_to_use):

            ### Image data path accumulation ###
            img_base_path = self.img_dataset_path + '/' + sequence_num + '/image_2'
            img_data_name = np.array(sorted(os.listdir(img_base_path)))

            ### Pose data accumulation ###
            lines = []
            pose_file = open(self.pose_dataset_path + '/' + sequence_num + '.txt', 'r')
            while True:
                line = pose_file.readline()
                
                if not line: break
                
                lines.append(line)
                
            pose_file.close()

            self.sequence_idx = 0
            pose_list = []

            ### Labeling using Agglomerative Clustering ###
            print('Agglomerative Clustering-based Labeling : Sequence : {}'.format(sequence_num))
            for line in np.array(lines):
                
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

                pose_list.append(np.array([current_x, current_y, current_z, current_roll, current_pitch, current_yaw]))

            pose_list = np.array(pose_list)
            xyz_pose_list = pose_list[:, 0:3]

            clusters = self.agglo_clusterer.fit(xyz_pose_list)

            ### Train, Valid, Test split dataset on each sequence ###
            cluster_labels = np.array(clusters.labels_)
            data_type_list = np.array(['train'] * len(cluster_labels))
            for label in np.unique(cluster_labels):

                mask = (cluster_labels == label)

                label_idx = np.where(mask)

                train_idx = []
                valid_idx = []
                test_idx = []
                valid_test_idx = []

                if len(label_idx[0]) > 1:
                    # Split between train and validation/test

                    train_idx, valid_test_idx = train_test_split(label_idx[0], test_size=test_ratio, random_state=42, shuffle=True)

                    data_type_list[train_idx] = 'train'

                    ### Create positive image path list for each cluster label ###
                    base_path_list = np.array([img_base_path] * len(train_idx))
                    path_list_append = np.array(['/'] * len(train_idx))
                    combined_path_list = np.core.defchararray.add(base_path_list, path_list_append)
                    combined_path_list = np.core.defchararray.add(combined_path_list, img_data_name[train_idx])
                    self.seq_path_dict['{}-{}'.format(sequence_num, label)] = combined_path_list

                else:
                    # Exception Handling (Only 1 image in cluster) : Randomly assign the leftover as validation or test
                    data_type_list[label_idx] = 'train'

                    ### Create positive image path list for each cluster label ###
                    base_path_list = np.array([img_base_path] * len(label_idx))
                    path_list_append = np.array(['/'] * len(label_idx))
                    combined_path_list = np.core.defchararray.add(base_path_list, path_list_append)
                    combined_path_list = np.core.defchararray.add(combined_path_list, img_data_name[label_idx])
                    self.seq_path_dict['{}-{}'.format(sequence_num, label)] = combined_path_list

                if len(valid_test_idx) > 1:
                    # Split between validation and test
                    valid_test_ratio = valid_ratio + test_ratio
                    rearranged_test_ratio = 1 - (valid_ratio / valid_test_ratio)

                    valid_idx, test_idx = train_test_split(valid_test_idx, test_size=rearranged_test_ratio, random_state=42, shuffle=True)

                    data_type_list[valid_idx] = 'valid'

                    data_type_list[test_idx] = 'test'

                else:
                    # Exception Handling (Only 1 image to split between validation and test) : Randomly assign the leftover as validation or test
                    for idx in valid_test_idx:

                        if random.random() >= 0.5:
                            data_type_list[idx] = 'valid'
                        else:
                            data_type_list[idx] = 'test'

            ### Writing dataset attributes on Full Dataset csv ###
            for img_name, pose, cluster_label, data_type in zip(img_data_name, pose_list, cluster_labels, data_type_list):
                
                data = [self.data_idx, sequence_num, self.sequence_idx, \
                        img_base_path + '/' + img_name, \
                        pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], \
                        '{}-{}'.format(sequence_num, cluster_label), \
                        data_type]

                self.full_dataset_writer.writerow(data)

                if data_type == 'train':
                    self.train_dataset_writer.writerow(data)
                
                elif data_type == 'valid':
                    self.valid_dataset_writer.writerow(data)

                elif data_type == 'test':
                    self.test_dataset_writer.writerow(data)

                self.data_idx += 1
                self.sequence_idx += 1

        self.full_dataset_dict.close()
        self.train_dataset_dict.close()
        self.valid_dataset_dict.close()
        self.test_dataset_dict.close()

class sensor_dataset(torch.utils.data.Dataset):

    def __init__(self, img_dataset_path='', pose_dataset_path='',
                       sequence_to_use=['00'], train_ratio=0, valid_ratio=0, test_ratio=0,
                       cluster_linkage='ward', cluster_distance=1.0, 
                       mode='training', output_resolution=[1280, 240],
                       transform=None):

        self.mode = mode
        self.output_resolution = output_resolution
        self.transform = transform

        if self.transform is None:
            self.transform = transforms.Compose([transforms.Resize((output_resolution[1], output_resolution[0])),
                                                 transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5)], p=0.5),
                                                 transforms.ToTensor(),
                                                 transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 0.3)),])
        else:
            self.transform = transform

        self.dataset_dict_generator = dataset_dict_generator(img_dataset_path=img_dataset_path, pose_dataset_path=pose_dataset_path,
                                                             sequence_to_use=sequence_to_use, train_ratio=train_ratio, valid_ratio=valid_ratio, test_ratio=test_ratio,
                                                             cluster_linkage=cluster_linkage, cluster_distance=cluster_distance)

        self.train_data_list = []
        train_dataset_dict = open(self.dataset_dict_generator.train_dataset_dict_path, 'r', encoding='utf-8')
        train_reader = csv.reader(train_dataset_dict)
        next(train_reader)     # Skip header row
        for row_data in train_reader:
            self.train_data_list.append(row_data)
        self.dataset_dict_generator.train_len = len(self.train_data_list)
        print('Training Length : {}'.format(self.dataset_dict_generator.train_len))
        train_dataset_dict.close()

        self.valid_data_list = []
        valid_dataset_dict = open(self.dataset_dict_generator.valid_dataset_dict_path, 'r', encoding='utf-8')
        valid_reader = csv.reader(valid_dataset_dict)
        next(valid_reader)     # Skip heaer row
        for row_data in valid_reader:
            self.valid_data_list.append(row_data)
        self.dataset_dict_generator.valid_len = len(self.valid_data_list)
        print('Validation Length : {}'.format(self.dataset_dict_generator.valid_len))
        valid_dataset_dict.close()

        self.test_data_list = []
        test_dataset_dict = open(self.dataset_dict_generator.test_dataset_dict_path, 'r', encoding='utf-8')
        test_reader = csv.reader(test_dataset_dict)
        next(test_reader)      # Skip header row
        for row_data in test_reader:
            self.test_data_list.append(row_data)
        self.dataset_dict_generator.test_len = len(self.test_data_list)
        print('Test Length : {}'.format(self.dataset_dict_generator.test_len))
        test_dataset_dict.close()

    def __getitem__(self, index):

        if self.mode == 'training':
            item = self.train_data_list[index]

        elif self.mode == 'validation':
            item = self.valid_data_list[index]

        elif self.mode == 'test':
            item = self.test_data_list[index]

        ##############################
        ### Anchor Image Selection ###
        ##############################

        anchor_img = np.array(Image.open(item[3]))
        anchor_img = cv.cvtColor(anchor_img, cv.COLOR_RGB2BGR)
        anchor_img = cv.resize(anchor_img, dsize=(self.output_resolution[0], self.output_resolution[1]), interpolation=cv.INTER_CUBIC)
        anchor_img = TF.to_tensor(anchor_img)
        

        ################################
        ### Positive Image Selection ###
        ################################

        anchor_img_label = item[10]

        positive_path_list = self.dataset_dict_generator.seq_path_dict[anchor_img_label]
        
        ### Exception Handling ###
        # If there are not enough images in the cluster, apply user-defined data augmentation.
        # Use augmented image as positive data
        if len(positive_path_list) <= 1:
            
            positive_img = Image.open(item[3])
            positive_img = self.transform(positive_img)

        else:
            
            positive_img_path = random.choice(positive_path_list)

            positive_img = np.array(Image.open(positive_img_path))
            positive_img = cv.cvtColor(positive_img, cv.COLOR_RGB2BGR)
            positive_img = cv.resize(positive_img, dsize=(self.output_resolution[0], self.output_resolution[1]), interpolation=cv.INTER_CUBIC)
            positive_img = TF.to_tensor(positive_img)

        ################################
        ### Negative Image Selection ###
        ################################

        negative_labels = list(self.dataset_dict_generator.seq_path_dict.keys())
        negative_labels.remove(anchor_img_label)

        negative_label_choice = random.choice(negative_labels)

        negative_path_list = self.dataset_dict_generator.seq_path_dict[negative_label_choice]
        
        negative_img_path = random.choice(negative_path_list)
        
        negative_img = np.array(Image.open(negative_img_path))
        negative_img = cv.cvtColor(negative_img, cv.COLOR_RGB2BGR)
        negative_img = cv.resize(negative_img, dsize=(self.output_resolution[0], self.output_resolution[1]), interpolation=cv.INTER_CUBIC)
        negative_img = TF.to_tensor(negative_img)
        
        return anchor_img, positive_img, negative_img

    def __len__(self):

        if self.mode == 'training':
            return self.dataset_dict_generator.train_len

        elif self.mode == 'validation':
            return self.dataset_dict_generator.valid_len

        elif self.mode == 'test':
            return self.dataset_dict_generator.test_len

