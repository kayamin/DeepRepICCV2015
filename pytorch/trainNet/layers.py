import os
import sys
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import pdb

# 畳み込み -> MaxPooling -> ReLU を行うクラス
# Live Repetition Counting ネットワーク

class RepetitionCountingNet(nn.Module):

    def __init__(self):
        super(RepetitionCountingNet, self).__init__()

        # takes input as Batch x Frames(channel) x Height x Width form
        convLayers = [

        nn.Conv2d(20, 40, 5, 1, bias=False),     # Bx20x50x50 -> Bx40x46x46
        nn.MaxPool2d(2),                        # Bx40x46x46 -> Bx40x23x23
        nn.ReLU(),
        nn.Conv2d(40, 60, 3, 1, bias=False),     # Bx40x23x23 -> Bx60x21x21
        nn.MaxPool2d(2),                        # Bx60x21x21 -> Bx60x10x10
        nn.ReLU(),
        nn.Conv2d(60, 90, 3, 1, bias=False),     # Bx60x10x10 -> Bx90x8x8
        nn.MaxPool2d(2),                        # Bx90x8x8   -> Bx90x4x4
        nn.ReLU(),

        ]

        fcLayers = [

        nn.Linear(4*4*90, 500, bias=False),  # Bx(4*4*90) -> Bx500
        nn.Linear(500, 8, bias=False)        # Bx500 -> Bx8

        ]

        self.convLayers = nn.Sequential(*convLayers)
        self.fcLayers = nn.Sequential(*fcLayers)

        # 重みの初期化（github実装にならって）
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

                m.weight.data.uniform_(-W_bound, W_bound)
                # m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                # m.weight.data.zero_()
                # m.bias.data.zero_()

    def forward(self, input):

        x = self.convLayers(input)
        x = x.view(-1, 4*4*90)
        x = self.fcLayers(x)

        return x
