'''
live repetition counting system
Ofir Levy, Lior Wolf
Tel Aviv University
'''
import os
import numpy
import cv2
import pickle
from common import *
import cortex.count


def get_inter_num(data, valid):
    (ns_test_set_x_st2,ns_test_set_x_st5,ns_test_set_x_st8) = data
    (valid_st2,valid_st5,valid_st8) = valid
    if (valid_st8 == 1):
        return ((ns_test_set_x_st8.shape[0]-6)/5)
    if (valid_st5 == 1):
        return ((ns_test_set_x_st8.shape[0]-21)/8)
    else:
        return ((ns_test_set_x_st8.shape[0]-81)/20)


def load_movie_data(fileName, localization_method, segmentation_path=None):

    (ns_test_set_x_st2, valid_st2) = load_next_test_data_wrapper(fileName, 2, localization_method, segmentation_path)
    (ns_test_set_x_st5, valid_st5) = load_next_test_data_wrapper(fileName, 5, localization_method, segmentation_path)
    (ns_test_set_x_st8, valid_st8) = load_next_test_data_wrapper(fileName, 8, localization_method, segmentation_path)

    return ((ns_test_set_x_st2,ns_test_set_x_st5,ns_test_set_x_st8), (valid_st2,valid_st5,valid_st8))


def count_in_interval(classify, test_set_x, ns_test_set_x, frame_residue, start, end):

    assert start <= end
    if (start == end):
        return (0, 0, 0)

    test_set_x.set_value(ns_test_set_x, borrow=True)

    rep_counts = 0
    entropy = 0
    for i in range(start,end):
        output_label , pYgivenX = classify(i)
        pYgivenX[pYgivenX==0] = numpy.float32(1e-30) # hack to output valid entropy
        entropy = entropy - (pYgivenX*numpy.log(pYgivenX)).sum()
        output_label = output_label + 3 # moving from label to cycle length
        if (i == 0):
            rep_counts = 20 / output_label
            frame_residue = 20 % output_label
        else:
            frame_residue += 1
            if (frame_residue >= output_label):
                rep_counts += 1;
                frame_residue = 0;

    ave_entropy = entropy/(end-start)
    return (rep_counts, frame_residue, ave_entropy)



def initial_count(classify, test_set_x, data, valid):

    (valid_st2,valid_st5,valid_st8) = valid
    (ns_test_set_x_st2,ns_test_set_x_st5,ns_test_set_x_st8) = data

    # classify st_2 it is always valid
    (st2_count, st2_res, st2_entropy) = count_in_interval(classify, test_set_x, ns_test_set_x_st2, 0, 0, 81)  #100 - 19 etc.

    # check if st5 is valid. if not return st2 count
    if (valid_st5 == 1):
        (st5_count, st5_res, st5_entropy) = count_in_interval(classify, test_set_x, ns_test_set_x_st5, 0, 0, 21)
    else:
        st8_entropy = numpy.inf

    if (valid_st8 == 1):
        (st8_count, st8_res, st8_entropy) = count_in_interval(classify, test_set_x, ns_test_set_x_st8, 0, 0, 6)
    else:
        st8_entropy = numpy.inf


    winner = numpy.nanargmin(numpy.array([st2_entropy, st5_entropy, st8_entropy]))

    if (winner == 0):
        # winner is stride 2
        return (st2_count, (st2_res*2/2,st2_res*2/5, st2_res*2/8))
    if (winner == 1):
        # winner is stride 5
        return (st5_count, (st5_res*5/2,st5_res*5/5, st5_res*5/8))
    if (winner == 2):
        # winner is stride 8
        return (st8_count, (st8_res*8/2,st8_res*8/5, st8_res*8/8))



def get_next_count(classify, test_set_x, data, valid, global_count, curr_residue, start_frame):

    (valid_st2,valid_st5,valid_st8) = valid
    (ns_test_set_x_st2,ns_test_set_x_st5,ns_test_set_x_st8) = data
    (curr_residue_st2, curr_residue_st5, curr_residue_st8) = curr_residue

    # classify st_2 it is always valid
    (st2_count, st2_res, st2_entropy) = count_in_interval(classify, test_set_x, ns_test_set_x_st2, curr_residue_st2, (start_frame/2-19), (start_frame/2-19)+20)
    # check if st5 is valid. if not return st2 count
    if (valid_st5 == 1):
        (st5_count, st5_res, st5_entropy) = count_in_interval(classify, test_set_x, ns_test_set_x_st5, curr_residue_st5, (start_frame/5-19), (start_frame/5-19)+8)
    else:
        st5_entropy = numpy.inf

    if (valid_st8 == 1):
        (st8_count, st8_res, st8_entropy) = count_in_interval(classify, test_set_x, ns_test_set_x_st8, curr_residue_st8, (start_frame/8-19), (start_frame/8-19)+5)
    else:
        st8_entropy = numpy.inf

    winner = numpy.nanargmin(numpy.array([st2_entropy, st5_entropy, st8_entropy]))

    if (winner == 0):
        # winner is stride 2
        return (global_count + st2_count, (st2_res*2/2,st2_res*2/5, st2_res*2/8))
    if (winner == 1):
        # winner is stride 5
        return (global_count + st5_count, (st5_res*5/2,st5_res*5/5, st5_res*5/8))
    if (winner == 2):
        # winner is stride 8
        return (global_count + st8_count, (st8_res*8/2,st8_res*8/5, st8_res*8/8))

def get_remain_count(classify, test_set_x, data, valid, global_count, curr_residue, start_frame):

    (valid_st2,valid_st5,valid_st8) = valid
    (ns_test_set_x_st2,ns_test_set_x_st5,ns_test_set_x_st8) = data
    (curr_residue_st2, curr_residue_st5, curr_residue_st8) = curr_residue

    # classify st_2 it is always valid
    (st2_count, st2_res, st2_entropy) = count_in_interval(classify, test_set_x, ns_test_set_x_st2, curr_residue_st2, (start_frame/2-19), ns_test_set_x_st2.shape[0])
    # check if st5 is valid. if not return st2 count
    if (valid_st5 == 1):
        (st5_count, st5_res, st5_entropy) = count_in_interval(classify, test_set_x, ns_test_set_x_st5, curr_residue_st5, (start_frame/5-19), ns_test_set_x_st5.shape[0])
    else:
        st5_entropy = numpy.inf

    if (valid_st8 == 1):
        (st8_count, st8_res, st8_entropy) = count_in_interval(classify, test_set_x, ns_test_set_x_st8, curr_residue_st8, (start_frame/8-19), ns_test_set_x_st8.shape[0])
    else:
        st8_entropy = numpy.inf


    winner = numpy.nanargmin(numpy.array([st2_entropy, st5_entropy, st8_entropy]))

    if (winner == 0):
        # winner is stride 2
        return (global_count + st2_count)
    if (winner == 1):
        # winner is stride 5
        return (global_count + st5_count)
    if (winner == 2):
        # winner is stride 8
        return (global_count + st8_count)


def count_entire_movie(classify, test_set_x, data, valid, global_count, curr_residue, start_frame):

    (valid_st2,valid_st5,valid_st8) = valid
    (ns_test_set_x_st2,ns_test_set_x_st5,ns_test_set_x_st8) = data
    (curr_residue_st2, curr_residue_st5, curr_residue_st8) = curr_residue

    # classify st_2 it is always valid
    (st2_count, st2_res, st2_entropy) = count_in_interval(classify, test_set_x, ns_test_set_x_st2, curr_residue_st2, 0, ns_test_set_x_st2.shape[0])
    # check if st5 is valid. if not return st2 count
    if (valid_st5 == 1):
        (st5_count, st5_res, st5_entropy) = count_in_interval(classify, test_set_x, ns_test_set_x_st5, curr_residue_st5, 0, ns_test_set_x_st5.shape[0])
    else:
        st5_entropy = numpy.inf

    if (valid_st8 == 1):
        (st8_count, st8_res, st8_entropy) = count_in_interval(classify, test_set_x, ns_test_set_x_st8, curr_residue_st8, 0, ns_test_set_x_st8.shape[0])
    else:
        st8_entropy = numpy.inf

    winner = numpy.nanargmin(numpy.array([st2_entropy, st5_entropy, st8_entropy]))

    if (winner == 0):
        # winner is stride 2
        return (global_count + st2_count)
    if (winner == 1):
        # winner is stride 5
        return (global_count + st5_count)
    if (winner == 2):
        # winner is stride 8
        return (global_count + st8_count)


def load_and_count_video(filename, classify, test_set_x, batch_size, localization_method, segmentation_path=None):

    # load all 3 stride for this movie
    (data, valid) = load_movie_data(filename, localization_method, segmentation_path)

    #workaround for short movies
    if data[0].shape[0] < 81:
        global_count = count_entire_movie(classify, test_set_x, data, valid, 0, (0,0,0), 0)
        return global_count

    # get initial counting. all 3 stride for 200 frames.
    # i.e. st8 runs 25 times. st5 runs 40 times. st2 runs 100 times
    (global_count, curr_residue) = initial_count(classify, test_set_x, data, valid)

    # get the last multiple of 40 global frame
    numofiterations = get_inter_num(data,valid)
    for start_frame in range(200, 200+(40*numofiterations), 40):
        # from now on sync every 40 frames.
        # i.e. st8 runs 5 times. st5 8 times and st2 20 times.
        (global_count, curr_residue) = get_next_count(classify, test_set_x, data, valid, global_count, curr_residue, start_frame)

    # for frames that left get from each
    global_count = get_remain_count(classify, test_set_x, data, valid, global_count, curr_residue, 200+(40*numofiterations))
    return global_count


def test_benchmark_online(classify, test_set_x, batch_size):

    strides = (2,5,8)
    #vid_root = "/home/trunia1/data/VideoCountingDataset/LevyWolf_Segments/videos/"
    #seg_root = "/home/trunia1/data/VideoCountingDataset/LevyWolf_Segments/localization/FastVideoSegment"

    vid_root = "/home/trunia1/data/VideoCountingDataset/LevyWolf_Segments/videos/"
    seg_root = None

    gt_counts = pickle.load( open( "vidGtData.p", "rb" ) )

    # This is the place where we set the localization method
    # Should be: 'full_frame', 'simple' or 'segmentation'
    localization_method = 'simple'

    gt = numpy.tile(gt_counts, (len(strides), 1))
    gt = gt.T
    gt1 = gt[:,1]
    tests_num = gt_counts.shape[0]
    predict = numpy.zeros(len(gt))

    for nTestSet in range(tests_num):

        fileName = vid_root+"YT_seg_{:02d}.avi".format(nTestSet)
        print(fileName)

        # Perform counting
        global_count = load_and_count_video(fileName, classify, test_set_x, batch_size, localization_method, seg_root)
        predict[nTestSet] = global_count

        print("  True Count = {}".format(gt1[nTestSet]))
        print("  Pred Count = {}".format(predict[nTestSet]))
        print("#"*60)


    output_dir = "/home/trunia1/experiments/2017/20170925_LevyWolf_FINAL/online/LevyWolf_Acceleration/{}/".format(localization_method)

    print("#"*60)
    gt1 = gt1.astype(numpy.int32)
    cortex.count.write_experiment(predict, gt1, output_dir)
