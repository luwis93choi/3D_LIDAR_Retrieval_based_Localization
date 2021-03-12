import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Function
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv

import math

import pytorch_ssim     # SSIM Loss : https://github.com/Po-Hsun-Su/pytorch-ssim

class CNN_Autoencoder(nn.Module):

    def __init__(self, device=None, input_size=[1, 3, 100, 100], batch_size=1, learning_rate=0.001):

        super(CNN_Autoencoder, self).__init__()

        # self.encoder = nn.Sequential(

        #     nn.Conv2d(in_channels=input_size[1], out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(0.1),

        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.1),

        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.1),

        #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(0.1),

        #     nn.Flatten(start_dim=1),
        # )

        ### Encoder ###        
        self.conv1 = nn.Conv2d(in_channels=input_size[1], out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.leakyrelu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchNorm2 = nn.BatchNorm2d(64)
        self.leakyrelu2 = nn.LeakyReLU(0.1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchNorm3 = nn.BatchNorm2d(128)
        self.leakyrelu3 = nn.LeakyReLU(0.1)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchNorm4 = nn.BatchNorm2d(256)
        self.leakyrelu4 = nn.LeakyReLU(0.1)

        ### Decoder ###
        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
        self.batchNorm_deconv1 = nn.BatchNorm2d(128)
        self.leakyrelu_deconv1 = nn.LeakyReLU(0.1)

        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
        self.batchNorm_deconv2 = nn.BatchNorm2d(64)
        self.leakyrelu_deconv2 = nn.LeakyReLU(0.1)

        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
        self.batchNorm_deconv3 = nn.BatchNorm2d(32)
        self.leakyrelu_deconv3 = nn.LeakyReLU(0.1)

        self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=input_size[1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # self.loss = nn.MSELoss()
        self.loss = pytorch_ssim.SSIM(window_size=11)

        self.batch_size = batch_size

        self.to(device)

    # CNN Layer Result Display Function - Display 2D Convolution Results by Channels
    def layer_disp(self, conv_result_x, window_name, col_num, resize_ratio=0.8, invert=False):

        x_disp = conv_result_x.clone().detach().cpu()

        if invert == True:
            img_stack = 255 - (x_disp.permute(0, 2, 3, 1)[0, :, :, :].numpy()*255).astype(np.uint8)
        else:
            img_stack = (x_disp.permute(0, 2, 3, 1)[0, :, :, :].numpy()*255).astype(np.uint8)

        channel = img_stack.shape[2]

        cols = col_num
        rows = int(math.ceil(channel/cols))

        resize_ratio = resize_ratio

        for i in range(rows):
            
            for j in range(cols):
                
                if (j + cols*i) >= channel:
                    blank = np.zeros((width, height, 1), np.uint8)
                    img_horizontal = cv.hconcat([img_horizontal, blank])

                elif j == 0:
                    img_horizontal = cv.resize(img_stack[:, :, j + cols*i], dsize=(0, 0), fx=resize_ratio, fy=resize_ratio, interpolation=cv.INTER_LINEAR)
                    width, height = img_horizontal.shape

                else:
                    input_img = cv.resize(img_stack[:, :, j + cols*i], dsize=(0, 0), fx=resize_ratio, fy=resize_ratio, interpolation=cv.INTER_LINEAR)
                    img_horizontal = cv.hconcat([img_horizontal, input_img])

            if i == 0:
                img_total = img_horizontal
            else:
                img_total = cv.vconcat([img_total, img_horizontal])

        cv.imshow(window_name, img_total)
        cv.waitKey(1)

    def forward(self, x):

        self.layer_disp(x, window_name='Input Img', col_num=2, resize_ratio=0.4)

        # print('----------------------------------------')
        # print(x.shape)

        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.leakyrelu1(x)

        # print(x.shape)

        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = self.leakyrelu2(x)

        # print(x.shape)

        x = self.conv3(x)
        x = self.batchNorm3(x)
        x = self.leakyrelu3(x)

        # print(x.shape)

        x = self.conv4(x)
        x = self.batchNorm4(x)
        x = self.leakyrelu4(x)

        # print(x.shape)

        x = self.deconv1(x)
        x = self.batchNorm_deconv1(x)
        x = self.leakyrelu_deconv1(x)

        # print(x.shape)

        x = self.deconv2(x)
        x = self.batchNorm_deconv2(x)
        x = self.leakyrelu_deconv2(x)

        # print(x.shape)

        x = self.deconv3(x)
        x = self.batchNorm_deconv3(x)
        x = self.leakyrelu_deconv3(x)

        # print(x.shape)

        x = self.deconv4(x)

        # print(x.shape)

        self.layer_disp(x, window_name='Recovered Img', col_num=2, resize_ratio=0.4)

        return x