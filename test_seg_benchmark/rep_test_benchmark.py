import cPickle
import os
import sys
import time
import numpy
import theano
import theano.tensor as T

from layers import LogisticRegression, HiddenLayer, LeNetConvPoolLayer
from common import load_initial_test_data
from bench_classify_online import test_benchmark_online
from bench_classify_offline import test_benchmark_offline


#---------------------------------------------------------------------------#


def prepare_network():

    rng = numpy.random.RandomState(23455)

    print('Preparing Theano model...')

    mydatasets = load_initial_test_data()
    test_set_x, test_set_y, shared_test_set_y, valid_ds = mydatasets
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]

    # allocate symbolic variables for the data
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    # image size
    layer0_w = 50
    layer0_h = 50
    layer1_w = (layer0_w-4)/2
    layer1_h = (layer0_h-4)/2
    layer2_w = (layer1_w-2)/2
    layer2_h = (layer1_h-2)/2
    layer3_w = (layer2_w-2)/2
    layer3_h = (layer2_h-2)/2

    ######################
    # BUILD NETWORK #
    ######################
    # image sizes
    batchsize     = 1
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

    layer0 = LeNetConvPoolLayer(
        rng, input=layer0_input,
        image_shape=signals_shape,
        filter_shape=filters_shape, poolsize=(2, 2))

    layer1 = LeNetConvPoolLayer(
        rng, input=layer0.output,
        image_shape=(batchsize, flt_channels, layer1_w, layer1_h),
        filter_shape=(60, flt_channels, 3, 3), poolsize=(2, 2))


    layer2 = LeNetConvPoolLayer(
        rng, input=layer1.output,
        image_shape=(batchsize, 60, layer2_w, layer2_h),
        filter_shape=(90, 60, 3, 3), poolsize=(2, 2))
    layer3_input = layer2.output.flatten(2)


    layer3 = HiddenLayer(
        rng, input=layer3_input, n_in=90 * layer3_w * layer3_h  ,
        n_out=500, activation=T.tanh)


    layer4 = LogisticRegression(input=layer3.output, n_in=500, n_out=8)


    cost = layer4.negative_log_likelihood(y)

    classify = theano.function([index], outputs=layer4.get_output_labels(y),
                               givens={
                                   x: test_set_x[index * batchsize: (index + 1) * batchsize],
                                   y: test_set_y[index * batchsize: (index + 1) * batchsize]})

    print('Loading network weights...')
    weightFile =  '../live_count/weights.save'
    f = open(weightFile, 'rb')
    loaded_objects = []
    for i in range(5):
        loaded_objects.append(cPickle.load(f))
    f.close()
    layer0.__setstate__(loaded_objects[0])
    layer1.__setstate__(loaded_objects[1])
    layer2.__setstate__(loaded_objects[2])
    layer3.__setstate__(loaded_objects[3])
    layer4.__setstate__(loaded_objects[4])

    return test_set_x, test_set_y, shared_test_set_y, valid_ds, classify, batchsize


if __name__ == '__main__':

    # Prepare the CNN in Theano
    test_set_x, test_set_y, shared_test_set_y, valid_ds, classify, batchsize = \
        prepare_network()

    if len(sys.argv) != 2:
        print('invalid arguments. Add --online for online entropy counting, --offline for offline entropy counting')
        sys.exit()
    if sys.argv[1] == "--online":
        print('online entropy counting start')
        test_benchmark_online(classify, test_set_x, batchsize)
    elif sys.argv[1] == "--offline":
        print('offline entropy counting start')
        test_benchmark_offline(classify, test_set_x, batchsize)
    else:
        raise ValueError("Please specify counting method as command line argument.")

    print("Done.")
