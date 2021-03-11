import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Function
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

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

        self.flatten_encoder = nn.Flatten(start_dim=1)
        
        self.encoder_linear = nn.Linear(in_features=256 * 15 * 80, out_features=2)


        ### Decoder ###
        self.decoder_linear = nn.Linear(in_features=2, out_features=256 * 15 * 80)

        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
        self.batchNorm_deconv1 = nn.BatchNorm2d(128)
        self.leakyrelu_deconv1 = nn.LeakyReLU(0.1)

        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
        self.batchNorm_deconv2 = nn.BatchNorm2d(64)
        self.leakyrelu_deconv2 = nn.LeakyReLU(0.1)

        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
        self.batchNorm_deconv3 = nn.BatchNorm2d(32)
        self.leakyrelu_deconv3 = nn.LeakyReLU(0.1)

        self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=6, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.loss = nn.MSELoss()

        self.batch_size = batch_size

        self.to(device)

    def forward(self, x):

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

        x = self.flatten_encoder(x)

        # print(x.shape)

        x = self.encoder_linear(x)

        # print(x.shape)

        x = self.decoder_linear(x)

        # print(x.shape)

        x = x.view(-1, 256, 15, 80)

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

        return x