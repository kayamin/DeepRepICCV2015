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

        convLayers = [

        nn.conv2d(20, 40, 5, 1, 1, bias=True),  # 50x50x20 -> 46x46x40
        nn.max_pool_2d(2),                      # 46x46x40 -> 23x23x40
        nn.ReLU(),
        nn.conv2d(40, 60, 3, 1, 1, bias=True),  # 23x23x40 -> 21x21x60
        nn.max_pool_2d(2),                      # 21x21x60 -> 10x10x60
        nn.ReLU(),
        nn.conv2d(60, 90, 3, 1, 1, bias=True),  # 10x10x60 -> 8x8x90
        nn.max_pool_2d(2),                      # 8x8x90 -> 4x4x90
        nn.ReLU(),
        nn.Linear(4*4*90, 500, bias=True),
        nn.Linear(500, 8, bias=True)

        ]

        self.convLayers = nn.Sequential(*convLayers)

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

                m.weight.data.uniform(-W_bound, W_bound)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                m.weight.data.zero_()
                m.bias.data.zero_()

    def forward(self, input):

        x = self.convLayers

        return x
