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

class CNN_Encoder(nn.Module):

    def __init__(self, device=None, input_size=[1, 3, 100, 100], batch_size=1, learning_rate=0.001, loss_type='ssim'):

        super(CNN_Encoder, self).__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(in_channels=input_size[1], out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        if loss_type is 'ssim':
            self.loss = pytorch_ssim.SSIM(window_size=11)

        elif loss_type is 'mse':
            self.loss = nn.MSELoss()

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

        self.layer_disp(x, window_name='Input Img', col_num=2, resize_ratio=1.0)

        # print('----------------------------------------')
        # print(x.shape)

        x = self.encoder(x)

        # print(x.shape)

        # self.layer_disp(x, window_name='Recovered Img', col_num=2, resize_ratio=1.0)

        return x