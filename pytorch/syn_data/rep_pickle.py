'''
live repetition counting system
Ofir Levy, Lior Wolf
Tel Aviv University
'''
import gzip
import os
import sys
import numpy as np
import pandas as pd
import h5py
import pdb

# main:


# matファイルからデータを取得
# cell 形式から読み取るのは若干特殊なので注意
# mat ファイル一つにつき (50trial x 20frame)x50x50 のデータが存在
# labels は 50trial 分

# 1trialずつ別のファイルに保存したほうが扱いやすい

def read_matdata(filename):
    with h5py.File(filename, 'r') as f:
        all_cFrames = list(f['all_cFrames'])
        labels = np.array(list(f['labels']))
        motion_types = np.array(list(f['motion_types']))

        syn_frames = np.zeros([len(all_cFrames), 50, 50])
        for i in range(len(all_cFrames)):
            b = all_cFrames[i][0]
            syn_frames[i, :, :] = np.array(list(f[b]))
    return [syn_frames, labels]


# main:

in_dir = '/mnt/ratmemory_hippocampus/kayama/AIL/out/mat/'
out_dir = '/mnt/ratmemory_hippocampus/kayama/AIL/out/h5/'

print("starting ...")

# prepare train set
trainset_list = []
for nSet in range(1,601):

    # load mat file
    filename_in = in_dir+'rep_train_data_' + str(nSet) + '.mat'
    syn_frames, labels = read_matdata(filename_in)
    syn_frames = syn_frames.reshape(-1, 20, 50, 50)

    # store in h5 file
    # 50trial とそのラベルをそれぞれ異なるファイルに保存

    for i in range(len(labels)):
        file_Idx = 50*(nSet-1) + i
        filename_out = out_dir+'rep_train_data_' + str(file_Idx) + '.gzip.h5'
        trainset_list.append(filename_out)
        file = h5py.File(filename_out)
        file.create_dataset('data_x',data=syn_frames[i],compression='gzip',compression_opts=9)
        file.create_dataset('data_y',data=labels[i],compression='gzip',compression_opts=9)
        file.close()
        print("done preparing train set number {}".format(file_Idx))

df = pd.DataFrame()
df['filename'] = trainset_list
df.to_csv('{}trainset_list.csv'.format(out_dir), index=False)

# prepare val set
validset_list = []
for nSet in range(1,101):

    # load mat file
    filename_in = in_dir+'rep_valid_data_' + str(nSet) + '.mat'
    syn_frames, labels = read_matdata(filename_in)
    syn_frames = syn_frames.reshape(-1, 20, 50, 50)
    
    # store in h5 file
    for i in range(len(labels)):
        file_Idx = 50*(nSet-1) + i
        filename_out = out_dir+'rep_valid_data_' + str(file_Idx) + '.gzip.h5'
        trainset_list.append(filename_out)
        file = h5py.File(filename_out)
        file.create_dataset('data_x',data=syn_frames[i],compression='gzip',compression_opts=9)
        file.create_dataset('data_y',data=labels[i],compression='gzip',compression_opts=9)
        file.close()

        print("done preparing validation set number {}".format(file_Idx))

df = pd.DataFrame()
df['filename'] = validset_list
df.to_csv('{}validset_list.csv'.format(out_dir), index=False)

print("done all")
