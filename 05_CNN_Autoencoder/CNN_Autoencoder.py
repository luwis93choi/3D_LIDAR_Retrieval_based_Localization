import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Function
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

class CNN_Autoencoder(nn.Module):

    def __init__(self, device=None, input_size=[3, 100, 100], batch_size=1, learning_rate=0.001):

        super(CNN_Autoencoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_size[0], out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.leakyrelu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchNorm2 = nn.BatchNorm2d(64)
        self.leakyrelu2 = nn.LeakyReLU(0.1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchNorm3 = nn.BatchNorm2d(128)
        self.leakyrelu3 = nn.LeakyReLU(0.1)

        # self.encoder_linear = nn.Linear()

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.to(device)

    def forward(self, x):

        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.leakyrelu1(x)

        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = self.leakyrelu2(x)

        x = self.conv3(x)
        x = self.batchNorm3(x)
        x = self.leakyrelu3(x)

        print(x.shape)

        return x