import os
import sys
import argparse
import cv2 as cv
import numpy as np
import open3d as o3d

import time
import matplotlib.pyplot as plt

import torch
from torch import threshold
from torch.utils.data import DataLoader

import collections

import pytorch_ssim     # SSIM Loss : https://github.com/Po-Hsun-Su/pytorch-ssim

from agg_cluster_dataset import sensor_dataset

from CNN_Autoencoder import CNN_Autoencoder

import torchvision.transforms.functional as TF

ap = argparse.ArgumentParser()

ap.add_argument('-i', '--input_img_file_path', type=str, required=True)
ap.add_argument('-p', '--input_pose_file_path', type=str, required=True)
ap.add_argument('-b', '--input_batch_size', type=int, required=True)
ap.add_argument('-c', '--input_CUDA_num', type=str, required=True)
ap.add_argument('-e', '--training_epoch', type=int, required=True)

args = vars(ap.parse_args())

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
                         train_ratio=0, valid_ratio=0, test_ratio=0,
                         cluster_linkage='ward', cluster_distance=1.0, 
                         mode='training', output_resolution=[1280, 240],
                         transform=None)

dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
dataloader.dataset.mode = 'training'

start_time = str(time.time())
loss_history = []
plt.figure(figsize=(10, 8))

loss_Q = collections.deque(maxlen=1000)

for epoch in range(training_epoch):

    # for batch_idx, (lidar_range_img_tensor, current_img_tensor, pose_6DOF_tensor) in enumerate(dataloader):
    for batch_idx, (current_img_tensor, pose_6DOF_tensor) in enumerate(dataloader):

        # combined_sensor_img_tensor = (torch.cat((lidar_range_img_tensor, current_img_tensor), dim=1)).to(PROCESSOR)
        combined_sensor_img_tensor = current_img_tensor.to(PROCESSOR)

        if (epoch == 0) and (batch_idx == 0):

            print('[Init Network]')

            Autoencoder = CNN_Autoencoder(device=PROCESSOR, input_size=combined_sensor_img_tensor.shape, batch_size=batch_size, learning_rate=0.001)

        Autoencoder.train()
        
        est_img_tensor = Autoencoder(combined_sensor_img_tensor)

        Autoencoder.optimizer.zero_grad()
        recovery_loss = -Autoencoder.loss(est_img_tensor, combined_sensor_img_tensor)
        recovery_loss.backward()
        Autoencoder.optimizer.step()

        loss_Q.append(recovery_loss.item())

        updates = []
        updates.append('\n')
        updates.append('[Train Epoch {}/{}][Progress : {:.2%}][Batch Idx : {}] \n'.format(epoch, training_epoch, batch_idx/len(dataloader), batch_idx))
        updates.append('[Immediate Loss] : {:.4f} \n'.format(recovery_loss.item()))
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
        f.write(str(recovery_loss.item()) + '\n')
        f.close()

        loss_history.append(sum(loss_Q) / len(loss_Q))
        plt.plot([i for i in range(len(loss_history))], loss_history, 'bo-')
        plt.title('CNN Autoencoder recovery training result - SSIM loss' + '\nSeq in Use : ' + str(seq_in_use))
        plt.xlabel('Iteration')
        plt.ylabel('Running Average SSIM loss')
        plt.tight_layout()
        plt.savefig('./' + start_time + '/Training_Result.png')
        plt.cla()

    torch.save({'epoch' : epoch,
                'Autoencoder' : Autoencoder.state_dict(),
                'Autoencoder_optimizer' : Autoencoder.optimizer.state_dict()}, './' + start_time + '/Autoencoder.pth')


