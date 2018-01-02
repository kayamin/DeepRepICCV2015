#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import time
import datetime
import h5py
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import pdb

from layers import RepetitionCountingNet
from ...util/Dataset_loader import Dataset_loader

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='DeepRepICCV2015')
    # learning & saving parameterss
    parser.add_argument('-lr', type=float, default=0.0002, help='initial learning rate [default: 0.0002]')
    parser.add_argument('-L1reg', type=float, default=0.0002, help='L1 regularization coefficient [default: 0.0002]')
    parser.add_argument('-L2reg', type=float, default=0.005, help='L2 regularization coefficient [default: 0.005]')
    parser.add_argument('-epochs', type=int, default=200, help='number of epochs for train [default: 200]')
    parser.add_argument('-batchsize', type=int, default=25, help='batch size for training [default: 25]')
    parser.add_argument('-savedir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('-savefreq', type=int, default=1, help='save learned model for every "-save-freq" epoch')
    parser.add_argument('-cuda', action='store_true', default=False, help='enable the gpu')
    # data souce
    parser.add_argument('-dataplace', type=str, default='/mnt/ratmemory_hippocampus/kayama/AIL/out/h5', help='prepared data path to run program')
    # model
    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot(snapshot/{Single or Multiple}/{date}/{epoch}) [default: None]')

    args = parser.parse_args()

    # update args and print
    args.save_dir = os.path.join(args.save_dir, 'TrainResult',datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    os.makedirs(args.save_dir)

    print("Parameters:")
    for attr, value in sorted(args.__dict__.items()):
        text ="\t{}={}\n".format(attr.upper(), value)
        print(text)
        with open('{}/Parameters.txt'.format(args.save_dir),'a') as f:
            f.write(text)


    # input data
    print('n\Loading data from [%s]...' % args.data_place)
    trainset_list_path = os.path.join(args.dataplace, 'trainset_list.csv')
    try:
        filename_df = pd.read_csv(args.dataplace)
    except:
        sprint("Sorry, failed to load data")

    # model
    D = RepetitionCountingNet()

    train_rep(filename_df, D, args)

    # else:
    #     # pose_code = [] # specify arbitrary pose code for every image
    #     pose_code = np.random.uniform(-1,1, (images.shape[0], Np))
    #     features = Generate_Image(images, pose_code, Nz, G, args)


def train_rep(filename_df, D_model, args):

    rng = np.random.RandomState(23455)

    train_dir = '../out/h5/'
    valid_dir = '../out/h5/'

    weights_dir = './weights/'

    print('... load input data')

    # create Network
    D_model = RepetitionCountingNet()
    if args.cuda:
        D_model.cuda()

    D_model.train()

    # parameter
    n_all_train_batches = 30000
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_all_train_batches /= batch_size
    n_train_batches /= batch_size
    n_valid_batches /= batch_size


    # train model
    print('... training')

    start_time = time.clock()

    data_set = Dataset_loader(filename_df)
