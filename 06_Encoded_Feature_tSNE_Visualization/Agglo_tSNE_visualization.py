import os
import sys
import argparse
import cv2 as cv
import numpy as np
import open3d as o3d

import time
import matplotlib
import matplotlib.pyplot as plt

import torch
from torch import threshold
from torch.utils.data import DataLoader

import collections

import pytorch_ssim     # SSIM Loss : https://github.com/Po-Hsun-Su/pytorch-ssim

from agg_cluster_dataset import sensor_dataset

from CNN_Encoder import CNN_Encoder

import torchvision.transforms.functional as TF

from sklearn.manifold import TSNE

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

ap = argparse.ArgumentParser()

ap.add_argument('-m', '--mode', type=str, required=True)    # Mode change : Model Training ('training') / t-SNE Visualization ('tsne')

ap.add_argument('-i', '--input_img_file_path', type=str, required=True)
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

dataset = sensor_dataset(img_dataset_path=args['input_img_file_path'], 
                         pose_dataset_path=args['input_pose_file_path'], 
                         sequence_to_use=seq_in_use, 
                         train_ratio=0.9, valid_ratio=0.05, test_ratio=0.05,
                         cluster_linkage='ward', cluster_distance=1.0, 
                         mode='training', output_resolution=[1280, 240],
                         transform=None)

if mode == 'training': 
    
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    dataloader.dataset.mode = 'training'

elif mode == 'tsne': 
    
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=True)
    dataloader.dataset.mode = 'training'

start_time = str(time.time())
loss_history = []
plt.figure(figsize=(10, 8))

loss_Q = collections.deque(maxlen=1000)

if mode == 'training':

    print('CNN Model training with Triplet Loss')

    for epoch in range(training_epoch):

        for batch_idx, (anchor_img, positive_img, negative_img, _) in enumerate(dataloader):

            anchor_img_tensor = anchor_img.to(PROCESSOR)
            positive_img_tensor = positive_img.to(PROCESSOR)
            negative_img_tensor = negative_img.to(PROCESSOR)

            if (epoch == 0) and (batch_idx == 0):

                print('[Init Network]')

                Feature_encoder = CNN_Encoder(device=PROCESSOR, input_size=anchor_img_tensor.shape, batch_size=batch_size, learning_rate=0.001, loss_type='mse')

            Feature_encoder.train()
            
            anchor_encoded_feature = Feature_encoder(anchor_img_tensor)
            positive_encoded_feature = Feature_encoder(positive_img_tensor)
            negative_encoded_feature = Feature_encoder(negative_img_tensor)
            margin = 1.0

            Feature_encoder.optimizer.zero_grad()
            triplet_loss = Feature_encoder.loss(anchor_encoded_feature, positive_encoded_feature) \
                        - Feature_encoder.loss(anchor_encoded_feature, negative_encoded_feature) \
                        + margin
            triplet_loss.backward()
            Feature_encoder.optimizer.step()

            loss_Q.append(triplet_loss.item())

            updates = []
            updates.append('\n')
            updates.append('[Train Epoch {}/{}][Progress : {:.2%}][Batch Idx : {}] \n'.format(epoch, training_epoch, batch_idx/len(dataloader), batch_idx))
            updates.append('[Immediate Loss] : {:.4f} \n'.format(triplet_loss.item()))
            updates.append('[Running Average Loss] : {:.4f} \n'.format(sum(loss_Q) / len(loss_Q)))
            final_updates = ''.join(updates)

            sys.stdout.write(final_updates)
            
            if batch_idx < len(dataloader)-1:
                for line_num in range(len(updates)):
                    sys.stdout.write("\x1b[1A\x1b[2K")


            if (epoch == 0) and (batch_idx == 0):
                if os.path.exists('./' + start_time) == False:
                    print('Creating save directory')
                    os.mkdir('./' + start_time)

            f = open('./' + start_time + '/running_avg_loss.txt', 'a')
            f.write(str(sum(loss_Q) / len(loss_Q)) + '\n')
            f.close()

            f = open('./' + start_time + '/immediate_avg_loss.txt', 'a')
            f.write(str(triplet_loss.item()) + '\n')
            f.close()

            loss_history.append(sum(loss_Q) / len(loss_Q))
            plt.plot([i for i in range(len(loss_history))], loss_history, 'bo-')
            plt.title('CNN Encoder - Triplet Loss' + '\nSeq in Use : ' + str(seq_in_use))
            plt.xlabel('Iteration')
            plt.ylabel('Running Average SSIM loss')
            plt.tight_layout()
            plt.savefig('./' + start_time + '/Training_Result.png')
            plt.cla()

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

            checkpoint = torch.load(model_path)

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
        encoded_feature_label_list.append(anchor_label)

        updates = []
        updates.append('\n')
        updates.append('[Progress : {:.2%}][Batch Idx : {}] \n'.format(batch_idx/len(dataloader), batch_idx))
        final_updates = ''.join(updates)

        sys.stdout.write(final_updates)

        if batch_idx < len(dataloader)-1:
            for line_num in range(len(updates)):
                sys.stdout.write("\x1b[1A\x1b[2K")



    ### Label color map generation ###

    num_clusters = np.unique(encoded_feature_label_list)

    color_map = []
    cmap = matplotlib.cm.get_cmap('rainbow')
    for i in range(num_clusters):
        color_map.append(cmap(i/num_clusters))
    color_map = np.array(color_map)

    

    ### t-SNE Encoded Feature Visualization ###

    encoded_feature_list = np.array(encoded_feature_list)

    print(encoded_feature_list.shape)

    embedded_encoded_features = TSNE(n_components=2, verbose=1, random_state=42, n_jobs=8).fit_transform(encoded_feature_list)

    print(embedded_encoded_features.shape)
        
    app = pg.mkQApp()

    win = pg.PlotWidget()
    win.resize(1000,600)
    win.setWindowTitle('CNN Autoencoder-based encoded feature visualization with t-SNE')
    win.show()

    scatter = pg.ScatterPlotItem(symbol='o', size=1)
    win.addItem(scatter)

    ptr = 0
    plot_len = len(embedded_encoded_features)

    def update():

        global win, scatter, ptr, color_map, embedded_encoded_features, plot_len

        embedded_features = np.transpose(embedded_encoded_features, (1, 0))

        pos = []

        plot_color_mask = color_map[encoded_feature_label_list[ptr]]

        plot_color = tuple(color_map[plot_color_mask])

        point = {'pos': embedded_features[:, ptr], 
                 'pen' : {'color' : plot_color, 'width' : 5}}
        
        pos.append(point)

        scatter.addPoints(pos)

        if ptr < plot_len-1:
            ptr += 1

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)

    app.exec_()
