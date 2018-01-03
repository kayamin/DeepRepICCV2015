import os
import sys
import numpy as np
import pickle

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import pdb

# 畳み込み -> MaxPooling -> ReLU を行うクラス
# Live Repetition Counting ネットワーク

class RepetitionCountingNet(nn.Module):

    def __init__(self, parameter_path=None):
        super(RepetitionCountingNet, self).__init__()

        # takes input as Batch x Frames(channel) x Height x Width form
        convLayers = [

        nn.Conv2d(20, 40, 5, 1, bias=True),     # Bx20x50x50 -> Bx40x46x46
        nn.MaxPool2d(2),                        # Bx40x46x46 -> Bx40x23x23
        nn.ReLU(),
        nn.Conv2d(40, 60, 3, 1, bias=True),     # Bx40x23x23 -> Bx60x21x21
        nn.MaxPool2d(2),                        # Bx60x21x21 -> Bx60x10x10
        nn.ReLU(),
        nn.Conv2d(60, 90, 3, 1, bias=True),     # Bx60x10x10 -> Bx90x8x8
        nn.MaxPool2d(2),                        # Bx90x8x8   -> Bx90x4x4
        nn.ReLU(),

        ]

        fcLayers = [

        nn.Linear(4*4*90, 500, bias=True),  # Bx(4*4*90) -> Bx500
        nn.Tanh(),
        nn.Linear(500, 8, bias=True)        # Bx500 -> Bx8

        ]

        self.convLayers = nn.Sequential(*convLayers)
        self.fcLayers = nn.Sequential(*fcLayers)
        self.softmaxLayer = nn.Softmax()

        # 重みの初期化
        if parameter_path:
            d = []
            with open(parameter_path, 'rb') as f :
                for i in range(5):
                    d.append(pickle.load(f, encoding='latin1'))

            cnt = 0;
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    weight = np.asarray(d[cnt][0])
                    bias = np.asarray(d[cnt][1])
                    m.weight.data = torch.FloatTensor(weight)
                    m.bias.data = torch.FloatTensor(bias)

                    cnt = cnt + 1

                if isinstance(m, nn.Linear):
                    weight = np.asarray(d[cnt][0]).transpose()
                    bias = np.asarray(d[cnt][1])
                    m.weight.data = torch.FloatTensor(weight)
                    m.bias.data = torch.FloatTensor(bias)
                    cnt = cnt + 1

        else:

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    in_ch = m.in_channels
                    out_ch = m.out_channels
                    FH = m.kernel_size[0]
                    FW = m.kernel_size[1]
                    Pooling_Size = 2

                    fan_in = in_ch * FH * FW
                    fan_out = out_ch * FH * FW / Pooling_Size
                    W_bound = np.sqrt(6. / (fan_in + fan_out))

                    nn.init.kaiming_normal(m.weight)
                    # m.weight.data.uniform_(-W_bound, W_bound)
                    # m.bias.data.zero_()

                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal(m.weight)
                    # m.weight.data.zero_()
                    # m.bias.data.zero_()

    def forward(self, input):

        x = self.convLayers(input)
        x = x.view(-1, 4*4*90)
        x = self.fcLayers(x)

        return x

    def get_output_labels(self, input):

        x = self.forward(input)
        self.p_y = self.softmaxLayer(x)
        _, self.y_pred = torch.max(self.p_y, 1)

        return [self.y_pred, self.p_y]
