import os
import sys
import argparse
import cv2 as cv
import numpy as np

import time
import datetime
import matplotlib
import matplotlib.pyplot as plt

import torch
from torch import threshold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import collections

import pytorch_ssim     # SSIM Loss : https://github.com/Po-Hsun-Su/pytorch-ssim

from agg_cluster_dataset import sensor_dataset

from CNN_Encoder import CNN_Encoder

import torchvision.transforms.functional as TF

from sklearn.manifold import TSNE

# import pyqtgraph as pg
# from pyqtgraph.Qt import QtGui, QtCore

from tqdm import tqdm

ap = argparse.ArgumentParser()

ap.add_argument('-m', '--mode', type=str, required=True)    # Mode change : Model Training ('training') / t-SNE Visualization ('tsne')

ap.add_argument('-i', '--input_lidar_file_path', type=str, required=True)
ap.add_argument('-p', '--input_pose_file_path', type=str, required=True)
ap.add_argument('-b', '--input_batch_size', type=int, required=True)
ap.add_argument('-c', '--input_CUDA_num', type=str, required=True)
ap.add_argument('-e', '--training_epoch', type=int, required=True)

ap.add_argument('-n', '--pretrained_model_path', type=str, required=False)

args = vars(ap.parse_args())

mode = args['mode']

batch_size = args['input_batch_size']

cuda_num = args['input_CUDA_num']

training_epoch = args['training_epoch']

if cuda_num != '':        
    # Load main processing unit for neural network
    PROCESSOR = torch.device('cuda:'+cuda_num if torch.cuda.is_available() else 'cpu')

print('Device in use : {}'.format(PROCESSOR))

seq_in_use = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

dataset = sensor_dataset(lidar_dataset_path=args['input_lidar_file_path'], 
                         pose_dataset_path=args['input_pose_file_path'], 
                         sequence_to_use=seq_in_use, 
                         train_ratio=0.9, valid_ratio=0.05, test_ratio=0.05,
                         cluster_linkage='ward', cluster_distance=1.0, 
                         mode='training', output_resolution=[1280, 240],
                         transform=None)

if mode == 'training': 
    
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True, prefetch_factor=20, persistent_workers=True)
    dataloader.dataset.mode = 'training'

elif mode == 'tsne': 
    
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=True)
    dataloader.dataset.mode = 'training'

start_time = str(datetime.datetime.now())
loss_history = []

loss_Q = collections.deque(maxlen=1000)

if mode == 'training':

    print('CNN Model training with Triplet Loss')

    training_writer = SummaryWriter(log_dir='./runs/' + start_time + '/3D_LiDAR_Img_Clustering')
    plot_step_training = 0

    train_learning_rate = 0.001

    for epoch in range(training_epoch):

        print('[Train Epoch {}]'.format(epoch))

        if epoch == 0:
            if os.path.exists('./' + start_time) == False:
                print('Creating save directory')
                os.mkdir('./' + start_time)

        for batch_idx, (anchor_img, positive_img, negative_img, _) in enumerate(tqdm(dataloader)):

            anchor_img_tensor = anchor_img.to(PROCESSOR)
            positive_img_tensor = positive_img.to(PROCESSOR)
            negative_img_tensor = negative_img.to(PROCESSOR)

            if (epoch == 0) and (batch_idx == 0):

                Feature_encoder = CNN_Encoder(device=PROCESSOR, input_size=anchor_img_tensor.shape, batch_size=batch_size, learning_rate=train_learning_rate, loss_type='ssim')

            Feature_encoder.train()
            
            anchor_encoded_feature = Feature_encoder(anchor_img_tensor)
            positive_encoded_feature = Feature_encoder(positive_img_tensor)
            negative_encoded_feature = Feature_encoder(negative_img_tensor)
            margin = 1.0

            Feature_encoder.optimizer.zero_grad()
            triplet_loss = Feature_encoder.positive_loss(anchor_encoded_feature, positive_encoded_feature) \
                           - Feature_encoder.negative_loss(anchor_encoded_feature, negative_encoded_feature) \
                           + margin
            triplet_loss.backward()
            Feature_encoder.optimizer.step()

            training_writer.add_scalar('Triplet Loss (SSIM) | Batch Size : {} | Learning Rate : {} | Optimizer : {}'.format(batch_size, train_learning_rate, type(Feature_encoder.optimizer)), triplet_loss.item(), plot_step_training)
            plot_step_training += 1

        torch.save({'epoch' : epoch,
                    'Autoencoder' : Feature_encoder.state_dict(),
                    'Autoencoder_optimizer' : Feature_encoder.optimizer.state_dict()}, './' + start_time + '/Autoencoder.pth')

elif mode == 'tsne':

    print('t-SNE CNN encoded feature visualization')

    encoded_feature_label_list = []
    encoded_feature_list = []
    for batch_idx, (anchor_img, _, _, anchor_label) in enumerate(dataloader):

        anchor_img_tensor = anchor_img.to(PROCESSOR)

        if batch_idx == 0:

            ### Model Loading ###
            model_path = args['pretrained_model_path']

            checkpoint = torch.load(model_path, map_location=PROCESSOR)

            if checkpoint == None:
                print('No Model loaded : {}'.format(model_path))
                
            else:
                Feature_encoder = CNN_Encoder(device=PROCESSOR, input_size=anchor_img_tensor.shape, batch_size=batch_size, learning_rate=0.001, loss_type='mse')
                Feature_encoder.load_state_dict(checkpoint['Autoencoder'])
                Feature_encoder.eval()

                print('Model loaded : {}'.format(args['pretrained_model_path']))

        encoded_feature_tensor = Feature_encoder(anchor_img_tensor)

        flat_encoded_feature_tensor = torch.flatten(encoded_feature_tensor, start_dim=1)
        flat_encoded_feature = flat_encoded_feature_tensor.clone().detach().cpu().numpy()[0]

        encoded_feature_list.append(flat_encoded_feature)
        encoded_feature_label_list.append(anchor_label[0])


    ### Label color map generation ###
    encoded_feature_label_list = np.array(encoded_feature_label_list)
    cluster_names = np.unique(encoded_feature_label_list)

    color_map = {}
    cmap = matplotlib.cm.get_cmap('rainbow')
    for cluster_name, i in zip(cluster_names, range(len(cluster_names))):
        color_map[cluster_name] = cmap(i/len(cluster_names))

    

    ### t-SNE Encoded Feature Visualization ###

    encoded_feature_list = np.array(encoded_feature_list)

    print(encoded_feature_list.shape)

    embedded_encoded_features = TSNE(n_components=2, verbose=2, random_state=42, n_jobs=1).fit_transform(encoded_feature_list)

    print(embedded_encoded_features.shape)
        
    # Use tensorboard embedding to visualizae feature clustering #
