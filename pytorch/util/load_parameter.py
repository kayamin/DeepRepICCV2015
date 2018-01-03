import numpy as np
import pickle

import torch
from torch.autograd import Variable

def load_parameter(D_model, parameter_path):

    d = []
    with open('weights.save', 'rb') as f :
        for i in range(5):
            d.append(pickle.load(f, encoding='latin1'))

    cnt = 0;
    for i, m in enumerate(D_model.modules()):
        if isinstance(m, nn.Conv2d):
            weight = np.asarray(d[cnt][0])
            bias = np.asarray(d[cnt][0])
            m.weight.data = torch.FloatTensor(weight)
            m.bias.data = torch.FloatTensor(bias)
            cnt = cnt + 1

        if isinstance(m, nn.Linear):
            weight = np.asarray(d[cnt][0]).transpose()
            bias = np.asarray(d[cnt][1])
            m.weight.data = torch.FloatTensor(weight)
            m.bias.data = torch.FloatTensor(bias)
            cnt = cnt + 1
