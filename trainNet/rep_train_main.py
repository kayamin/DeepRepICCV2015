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
import h5py
import theano
import theano.tensor as T

import scipy.ndimage
from scipy.io import matlab
from layers import LogisticRegression, HiddenLayer, LeNetConvPoolLayer



def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32'), shared_y


def load_initial_test_data():
    # just init gpu data with zeros
    data_x = numpy.zeros((5,50000), dtype=numpy.float)
    labels = numpy.zeros((1,5), dtype=numpy.uint8)
    data_x = data_x.astype(theano.config.floatX)
    data_y = labels
    data_y = data_y.reshape((data_y.shape[1]))
    data_y = data_y.astype(theano.config.floatX)
    train_set = data_x, data_y
    train_set_x, train_set_y, shared_train_set_y  = shared_dataset(train_set)
    rval = (train_set_x, train_set_y, shared_train_set_y, 1)
    return rval

def load_initial_data(filename):

    f = h5py.File(filename,'r')
    data_x = f['data_x'].value
    data_y = f['data_y'].value
    train_set = data_x, data_y
    f.close()
    train_set_x, train_set_y, shared_train_set_y = shared_dataset(train_set)

    rval = (train_set_x, train_set_y, shared_train_set_y)
    return rval



def load_next_data(filename):

    f = h5py.File(filename,'r')
    data_x = f['data_x'].value
    data_y = f['data_y'].value
    train_set = data_x, data_y
    f.close()
    return train_set

def train_rep(learning_rate=0.002, L1_reg=0.0002, L2_reg=0.005, n_epochs=200,
                    nkerns=[20, 50], batch_size=25):

    rng = numpy.random.RandomState(23455)

    train_dir = '../out/h5/'
    valid_dir = '../out/h5/'

    weights_dir = './weights/'

    print '... load input data'
    filename = train_dir+'rep_train_data_1.gzip.h5'
    datasets = load_initial_data(filename)
    train_set_x, train_set_y, shared_train_set_y = datasets

    filename = valid_dir+'rep_valid_data_1.gzip.h5'
    datasets = load_initial_data(filename)
    valid_set_x, valid_set_y, shared_valid_set_y = datasets


    mydatasets = load_initial_test_data()
    test_set_x, test_set_y, shared_test_set_y, valid_ds = mydatasets


    # compute number of minibatches for training, validation and testing
    n_all_train_batches = 30000
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_all_train_batches /= batch_size
    n_train_batches /= batch_size
    n_valid_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    # image size
    layer0_w = 50
    layer0_h = 50
    layer1_w = (layer0_w-4)/2 # 23
    layer1_h = (layer0_h-4)/2
    layer2_w = (layer1_w-2)/2 # 10.5
    layer2_h = (layer1_h-2)/2
    layer3_w = (layer2_w-2)/2 # 10.5 -> 10 に自動的になる？？
    layer3_h = (layer2_h-2)/2


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # image sizes
    batchsize     = batch_size
    in_channels   = 20
    in_width      = 50
    in_height     = 50
    #filter sizes
    flt_channels  = 40
    flt_time      = 20
    flt_width     = 5
    flt_height    = 5

    signals_shape = (batchsize, in_channels, in_height, in_width)
    filters_shape = (flt_channels, in_channels, flt_height, flt_width)

    layer0_input = x.reshape(signals_shape)

    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
                image_shape=signals_shape,
                filter_shape=filters_shape, poolsize=(2, 2))

    # TODO: incase of flt_time < in_time the output dimension will be different
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, flt_channels, layer1_w, layer1_h),
            filter_shape=(60, flt_channels, 3, 3), poolsize=(2, 2))


    layer2 = LeNetConvPoolLayer(rng, input=layer1.output,
                image_shape=(batch_size, 60, layer2_w, layer2_h),
                filter_shape=(90, 60, 3, 3), poolsize=(2, 2))
    layer3_input = layer2.output.flatten(2)


    layer3 = HiddenLayer(rng, input=layer3_input, n_in=90 * layer3_w * layer3_h  ,
                         n_out=500, activation=T.tanh)


    layer4 = LogisticRegression(input=layer3.output, n_in=500, n_out=8)

    classify = theano.function([index], outputs=layer4.get_output_labels(y),
                                  givens={
                                      x: test_set_x[index * batch_size: (index + 1) * batch_size],
                                      y: test_set_y[index * batch_size: (index + 1) * batch_size]})


    validate_model = theano.function([index], layer4.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # symbolic Theano variable that represents the L1 regularization term
    L1 = T.sum(abs(layer4.params[0])) + T.sum(abs(layer3.params[0])) + T.sum(abs(layer2.params[0])) + T.sum(abs(layer1.params[0])) + T.sum(abs(layer0.params[0]))
    # symbolic Theano variable that represents the squared L2 term
    L2_sqr = T.sum(layer4.params[0] ** 2) + T.sum(layer3.params[0] ** 2) + T.sum(layer2.params[0] ** 2) + T.sum(layer1.params[0] ** 2) + T.sum(layer0.params[0] ** 2)
    # the loss
    cost = layer4.negative_log_likelihood(y) + L1_reg * L1 + L2_reg * L2_sqr

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

    train_model = theano.function([index], cost, updates=updates,
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    start_time = time.clock()

    epoch = 0
    done_looping = False
    cost_ij = 0
    train_files_num = 600
    val_files_num = 100

    startc = time.clock()
    while (epoch < n_epochs) and (not done_looping):
        endc = time.clock()
        print('epoch %i, took %.2f minutes' % \
                                  (epoch, (endc - startc) / 60.))
        startc = time.clock()
        epoch = epoch + 1
        for nTrainSet in xrange(1,train_files_num+1):
            # load next train data
            if nTrainSet % 50 == 0:
                print 'training @ nTrainSet =  ', nTrainSet, ', cost = ',cost_ij
            filename = train_dir+'rep_train_data_' + str(nTrainSet) + '.gzip.h5'
            datasets = load_next_data(filename)
            ns_train_set_x, ns_train_set_y = datasets
            train_set_x.set_value(ns_train_set_x, borrow=True)
            shared_train_set_y.set_value(numpy.asarray(ns_train_set_y, dtype=theano.config.floatX), borrow=True)
            n_train_batches = train_set_x.get_value(borrow=True).shape[0]
            n_train_batches /= batch_size

            # train
            for minibatch_index in xrange(n_train_batches):

                # training itself
                # --------------------------------------
                cost_ij = train_model(minibatch_index)
                # -------------------------

        # at the end of each epoch run validation
        this_validation_loss = 0
        for nValSet in xrange(1,val_files_num+1):
            filename = valid_dir+'rep_valid_data_' + str(nValSet) + '.gzip.h5'
            datasets = load_next_data(filename)
            ns_valid_set_x, ns_valid_set_y = datasets
            valid_set_x.set_value(ns_valid_set_x, borrow=True)
            shared_valid_set_y.set_value(numpy.asarray(ns_valid_set_y, dtype=theano.config.floatX), borrow=True)
            n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
            n_valid_batches /= batch_size

            # compute zero-one loss on validation set
            validation_losses = [validate_model(i) for i
                                 in xrange(n_valid_batches)]
            this_validation_loss += numpy.mean(validation_losses)
        this_validation_loss /= (val_files_num)
        print('epoch %i, minibatch %i/%i, validation error %f %%' % \
              (epoch, minibatch_index + 1, n_train_batches, \
               this_validation_loss * 100.))


        # save snapshots
        print 'saving weights state, epoch = ', epoch
        f = file(weights_dir+'weights_epoch'+str(epoch)+'.save', 'wb')
        state_L0 = layer0.__getstate__();
        cPickle.dump(state_L0, f, protocol=cPickle.HIGHEST_PROTOCOL)
        state_L1 = layer1.__getstate__();
        cPickle.dump(state_L1, f, protocol=cPickle.HIGHEST_PROTOCOL)
        state_L2 = layer2.__getstate__();
        cPickle.dump(state_L2, f, protocol=cPickle.HIGHEST_PROTOCOL)
        state_L3 = layer3.__getstate__();
        cPickle.dump(state_L3, f, protocol=cPickle.HIGHEST_PROTOCOL)
        state_L4 = layer4.__getstate__();
        cPickle.dump(state_L4, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()


    end_time = time.clock()
    print('Optimization complete.')
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    train_rep()
