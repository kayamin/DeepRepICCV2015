#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import time
import datetime
import argparse
import h5py
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import pdb

sys.path.append('../')
from layers import RepetitionCountingNet
from util.Dataset_loader import Dataset_loader
from util.log_learning import log_learning

def main():

    parser = argparse.ArgumentParser(description='DeepRepICCV2015')
    # learning & saving parameterss
    parser.add_argument('-lr', type=float, default=0.0002, help='initial learning rate [default: 0.0002]')
    parser.add_argument('-L1reg', type=float, default=0.0002, help='L1 regularization coefficient [default: 0.0002]')
    parser.add_argument('-L2reg', type=float, default=0.005, help='L2 regularization coefficient [default: 0.005]')
    parser.add_argument('-epochs', type=int, default=200, help='number of epochs for train [default: 200]')
    parser.add_argument('-batchsize', type=int, default=25, help='batch size for training [default: 25]')
    parser.add_argument('-savedir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('-savefreq', type=int, default=1, help='save learned model for every "-savefreq" epoch')
    parser.add_argument('-cuda', action='store_true', default=False, help='enable the gpu')
    # data souce
    parser.add_argument('-dataplace', type=str, default='/mnt/ratmemory_hippocampus/kayama/AIL/out/h5', help='prepared data path to run program')
    # model
    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')

    args = parser.parse_args()

    # update args and print
    args.savedir = os.path.join(args.savedir,datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    os.makedirs(args.savedir)

    print("Parameters:")
    for attr, value in sorted(args.__dict__.items()):
        text ="\t{}={}\n".format(attr.upper(), value)
        print(text)
        with open('{}/Parameters.txt'.format(args.savedir),'a') as f:
            f.write(text)


    # input data
    print('n\Loading data from [%s]...' % args.dataplace)
    trainset_list_path = os.path.join(args.dataplace, 'trainset_list.csv')
    validset_list_path = os.path.join(args.dataplace, 'validset_list.csv')
    try:
        trainfilename_df = pd.read_csv(trainset_list_path)
        validfilename_df = pd.read_csv(validset_list_path)

    except:
        sprint("Sorry, failed to load data")

    # model
    D_model = RepetitionCountingNet(args.snapshot)

    train_rep(D_model, trainfilename_df, validfilename_df, args)


def train_rep(D_model, trainfilename_df, validfilename_df, args):

    rng = np.random.RandomState(23455)

    train_dir = '../out/h5/'
    valid_dir = '../out/h5/'

    weights_dir = './weights/'

    print('... load input data')

    # create Network
    D_model = RepetitionCountingNet()
    if args.cuda:
        D_model.cuda()

    # parameter

    # データは 50trial ずつ保存され，600ファイル , 30000 trial 存在
    # 3d のデータセットを 4次元方向 trial 毎にまとめたデータセットを用意する必要がある
    # 現状は 3d でまとまっているので 分解4d化 →　モデルに投入 が必要

    train_datasize = trainfilename_df.shape[0]

    optimizer_D = optim.SGD(D_model.parameters(), lr = args.lr)
    losss_criterion = nn.CrossEntropyLoss()

    # train model
    print('... training')
    loss_log = []
    steps = 0

    start_time = time.clock()

    for epoch in range(1, args.epochs+1):

        # train model

        D_model.train()
        data_set = Dataset_loader(trainfilename_df)
        dataloader = DataLoader(data_set, batch_size = args.batchsize, shuffle=True)

        for i, batch_data in enumerate(dataloader):
            D_model.zero_grad()

            batch_VideoBlocks = torch.FloatTensor(batch_data[0].float())
            batch_labels = torch.LongTensor(batch_data[1])
            l1_reg = torch.FloatTensor(1)
            l2_reg = torch.FloatTensor(1)

            if args.cuda:
                batch_VideoBlocks, batch_labels, l1_reg, l2_reg = \
                    batch_VideoBlocks.cuda(), batch_labels.cuda(), l1_reg.cuda(), l2_reg.cuda()

            batch_VideoBlocks, batch_labels = Variable(batch_VideoBlocks), Variable(batch_labels)
            l1_reg, l2_reg = Variable(l1_reg, requires_grad=True), Variable(l2_reg, requires_grad=True)


            # calcurate prediction and its cross entropy loss
            count_pred = D_model(batch_VideoBlocks)
            precision = calc_precision(count_pred, batch_labels)
            Loss_pred = losss_criterion(count_pred, batch_labels)

            # calcurate L1, L2 norm
            for W in D_model.parameters():
                l1_reg = l1_reg + W.norm(1)
                l2_reg = l2_reg + W.norm(2)


            # integrate all loss and do backpropagation, update parameters
            Loss = Loss_pred + args.L1reg * l1_reg + args.L2reg * l2_reg
            Loss.backward()
            optimizer_D.step()

            log_learning(epoch, steps, 'TrainLoss',  Loss_pred.data[0], args, precision)

            steps += 1


        # at the end of each epoch run validation

        D_model.eval()
        data_set = Dataset_loader(validfilename_df)
        dataloader = DataLoader(data_set, batch_size = args.batchsize, shuffle=True)

        valid_loss = []
        valid_precision = []
        for i, batch_data in enumerate(dataloader):
            batch_VideoBlocks = torch.FloatTensor(batch_data[0].float())
            batch_labels = batch_data[1]

            if args.cuda:
                batch_VideoBlocks, batch_labels = batch_VideoBlocks.cuda(), batch_labels.cuda()

            batch_VideoBlocks, batch_labels = Variable(batch_VideoBlocks), Variable(batch_labels)

            # calcurate prediction and its cross entropy loss
            count_pred = D_model(batch_VideoBlocks)
            precivion = calc_precision(count_pred, batch_labels)
            Loss_pred = losss_criterion(count_pred, batch_labels)

            valid_precision.append(precision)
            valid_loss.append(Loss_pred.data[0])

        valid_precision_mean = np.mean(valid_precision)
        valid_loss_mean = np.mean(valid_loss)

        log_learning(epoch, steps, 'ValidationLoss', valid_loss_mean, args, valid_precision_mean)

        # save snapshot
        if epoch%args.savefreq == 0:
            if not os.path.isdir(args.savedir): os.makedirs(args.savedir)
            save_path_D = os.path.join(args.savedir,'epoch{}_D.pt'.format(epoch))
            torch.save(D_model, save_path_D)



def calc_precision(count_pred, batch_labels):

    _, pred_labels = torch.max(count_pred, 1)

    precision = (pred_labels==batch_labels).type(torch.FloatTensor).sum() / count_pred.size()[0]

    # Variable(FloatTensor) -> Float へと変換
    precision = precision.data[0]

    return precision


if __name__=="__main__":
    main()
