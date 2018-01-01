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

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='DeepRepICCV2015')
    # learning & saving parameterss
    parser.add_argument('-lr', type=float, default=0.0002, help='initial learning rate [default: 0.0002]')
    parser.add_argument('-L1reg', type=float, default=0.0002, help='L1 regularization coefficient [default: 0.0002]')
    parser.add_argument('-L2reg', type=float, default=0.005, help='L2 regularization coefficient [default: 0.005]')
    parser.add_argument('-epochs', type=int, default=200, help='number of epochs for train [default: 200]')
    parser.add_argument('-batch-size', type=int, default=25, help='batch size for training [default: 25]')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('-save-freq', type=int, default=1, help='save learned model for every "-save-freq" epoch')
    parser.add_argument('-cuda', action='store_true', default=False, help='enable the gpu')
    # data souce
    parser.add_argument('-data-place', type=str, default='./data', help='prepared data path to run program')
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
    try:
        image_attributes_df, Nd, Np, Ni, Nz, channel_num = DataLoader(args.data_place)
    except:
        print("Sorry, failed to load data")

    # model
    D = RepetitionCountingNet()

    if not(args.generate):
        if not(args.multi_DRGAN):
            train_single_DRGAN(image_attributes_df, Nd, Np, Ni, Nz, D, G, args)
        else:
            if args.batch_size % args.images_perID == 0:
                train_multiple_DRGAN(image_attributes_df, Nd, Np, Ni, Nz, D, G, args)
            else:
                print("Please give valid combination of batch_size, images_perID")
                exit()
    # else:
    #     # pose_code = [] # specify arbitrary pose code for every image
    #     pose_code = np.random.uniform(-1,1, (images.shape[0], Np))
    #     features = Generate_Image(images, pose_code, Nz, G, args)


def train_rep():

    rng = np.random.RandomState(23455)

    train_dir = '../out/h5/'
    valid_dir = '../out/h5/'

    weights_dir = './weights/'

    print('... load input data')

    # create Network
    D = RepetitionCountingNet()

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
