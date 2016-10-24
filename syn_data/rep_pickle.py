'''
live repetition counting system
Ofir Levy, Lior Wolf
Tel Aviv University
'''
import cPickle
import gzip
import os
import sys
import time
import numpy
import scipy
from scipy.io import matlab
import theano
import theano.tensor as T
import h5py


#---------------------------------------------------------------------------#

def load_single_set(cFrames, set_num):
    
    datset = []
    for i in range(0,20):
        xx = cFrames[0,20*set_num+i]
        datset.append(xx)
    npdataset = numpy.array(datset, ndmin=4)
    return npdataset

#---------------------------------------------------------------------------#

def load_rep_dataset(filename):
    
    mat = matlab.loadmat(filename)
    cFrames = mat['all_cFrames']
    labels = mat['labels']
    labels = labels
    n_sets = labels.shape[1]
    # load input data
    data_x = load_single_set(cFrames, 0)
    for i in xrange(1,n_sets):
        data_x = numpy.append(data_x,load_single_set(cFrames, i),axis=0)    
    data_x = data_x.reshape((data_x.shape[0],data_x.shape[1]*data_x.shape[2]*data_x.shape[3]))
    data_x = data_x.astype(numpy.float32)
    data_y = labels
    data_y = data_y.reshape((data_y.shape[1]))
    data_y = data_y.astype(theano.config.floatX)
    train_set = data_x , data_y
    return train_set


#---------------------------------------------------------------------------#

# main:

in_dir = '../out/mat/'
out_dir = '../out/h5/'

print "starting ..."

# prepare train set
for nSet in range(1,601):
        
    # load mat file    
    filename = in_dir+'rep_train_data_' + str(nSet) + '.mat'
    train_set = load_rep_dataset(filename)
    
    # store in h5 file
    filename = out_dir+'rep_train_data_' + str(nSet) + '.gzip.h5'
    file = h5py.File(filename)
    data_x , data_y = train_set
    file.create_dataset('data_x',data=data_x,compression='gzip',compression_opts=9)
    file.create_dataset('data_y',data=data_y,compression='gzip',compression_opts=9)  
    file.close()
    print "done preparing train set number " , nSet

# prepare val set
for nSet in range(1,101):
        
    # load mat file
    filename = in_dir+'rep_valid_data_' + str(nSet) + '.mat'    
    train_set = load_rep_dataset(filename)
    
    # store in h5 file
    filename = out_dir+'rep_valid_data_' + str(nSet) + '.gzip.h5'
    file = h5py.File(filename)
    data_x , data_y = train_set
    file.create_dataset('data_x',data=data_x,compression='gzip',compression_opts=9)
    file.create_dataset('data_y',data=data_y,compression='gzip',compression_opts=9)    
    file.close()

    print "done preparing validation set number " , nSet

print "done all"
