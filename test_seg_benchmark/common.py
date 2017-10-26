'''
live repetition counting system
Ofir Levy, Lior Wolf
Tel Aviv University
'''
import os
import numpy as np
import theano
import theano.tensor as T
from scipy.ndimage import filters
import scipy
import cv2

import cortex.utils
from cortex.vision.video_reader import VideoReaderOpenCV
import cortex.vision.fast_seg as fast_seg


def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32'), shared_y

def load_single_set(cFrames, set_num):

    xx = cFrames[0,20*set_num+0]
    datset = xx.reshape(1,xx.shape[0]*xx.shape[1])

    for i in range(1,20):
        xx = cFrames[0,20*set_num+i]
        xx = xx.reshape(1,xx.shape[0]*xx.shape[1])
        datset = np.append(datset, xx, axis=1)
    return datset


def load_initial_test_data():
    # just init gpu data with zeros
    data_x = np.zeros((5,50000), dtype=np.float)
    labels = np.zeros((1,5), dtype=np.uint8)
    data_x = data_x.astype(theano.config.floatX)
    data_y = labels
    data_y = data_y.reshape((data_y.shape[1]))
    data_y = data_y.astype(theano.config.floatX)
    train_set = data_x, data_y
    train_set_x, train_set_y, shared_train_set_y  = shared_dataset(train_set)
    rval = (train_set_x, train_set_y, shared_train_set_y, 1)
    return rval


def get_boundingbox(frame_set, use_region_of_interest=False):

    fstd = np.std(frame_set,axis=0)
    framesstd = np.mean(fstd)
    #th = framesstd  / 3
    th = framesstd
    #ones = np.ones(8)
    ones = np.ones(10)
    big_var = (fstd>th)

    if not use_region_of_interest or framesstd==0:
        # no bb, take full frame
        frameROIRes = np.zeros([20,50,50])
        for i in range(20):
            frameROIRes[i,:,:] = scipy.misc.imresize(frame_set[i,:,:], size=(50,50),interp='bilinear')
        #frameROIRes = np.reshape(frameROIRes, (1,frameROIRes.shape[0]*frameROIRes.shape[1]*frameROIRes.shape[2]))
        frameROIRes = frameROIRes.astype(np.float32)
        return frameROIRes  #, framesstd)

    big_var = big_var.astype(np.float32)
    big_var = filters.convolve1d(big_var, ones, axis=0)
    big_var = filters.convolve1d(big_var, ones, axis=1)

    th2 = 80
    i,j = np.nonzero(big_var>th2)

    if (i.size > 0):

        si = np.sort(i)
        sj = np.sort(j)


        ll = si.shape[0]
        th1 = int(round(ll*0.03))
        th2 = int(np.floor(ll*0.98))

        y1 = si[th1]
        y2 = si[th2]
        x1 = sj[th1]
        x2 = sj[th2]

        # cut image ROI
        if (((x2-x1)>0) and ((y2-y1)>0)):
            framesRoi = frame_set[:,y1:y2,x1:x2]
        else:
            framesRoi = frame_set[:,:,:]
    else:
        framesRoi = frame_set[:,:,:]

    # debug - show ROI
    #cv2.namedWindow('ROI', cv2.WINDOW_NORMAL)
    #bla= scipy.misc.imresize(framesRoi[19,:,:], size=(200,200),interp='bilinear')
    #cv2.imshow('ROI', bla)

    # resize to 50x50
    frameROIRes = np.zeros([20,50,50])
    for i in range(20):
        frameROIRes[i,:,:] = scipy.misc.imresize(framesRoi[i,:,:], size=(50,50),interp='bilinear')

    #frameROIRes = frameROIRes / 255  # TODO - does this really nessacarry?
    return (frameROIRes)


def load_next_test_data_wrapper(filename, stride, localization_method, segmentation_path=None):

    assert localization_method in ['full_frame', 'simple', 'segmentation']

    if localization_method == 'full_frame':
        # Return frame block WITHOUT simple region of interest method
        return load_next_test_data_simple_roi(filename, stride, False)
    elif localization_method == 'simple':
        # Return frame block WITH simple region of interest method
        return load_next_test_data_simple_roi(filename, stride, True)
    else:
        # Return frame blocks WITH more advanced segmentation masks
        assert segmentation_path is not None, "Need segmentation path..."
        return load_next_test_data_segmentation(filename, segmentation_path, stride)

def load_next_test_data_simple_roi(fileName, stride, use_region_of_interest=True):

    cap = cv2.VideoCapture(fileName)
    frm_cnt = -1

    assert os.path.exists(fileName)
    assert cap.isOpened()

    #stride = 8
    framesList = []
    framesData = []

    while True:
        ret, frame = cap.read()
        if (ret == 0):
            break
        frm_cnt = frm_cnt + 1
        # take every nth frame
        if (frm_cnt%stride != 0):
            continue
        # convert to gray
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # sub-sample for performance
        gray_frame = gray_frame[::3,::3]
        # append to group
        framesList.append(gray_frame)
        if (len(framesList)>20):
            framesList.pop(0)
            # send for bounding box
            framesArr = np.array(framesList)
            frames = get_boundingbox(framesArr, use_region_of_interest)
            # append sequence
            frames = np.reshape(frames, (1,frames.shape[0]*frames.shape[1]*frames.shape[2]))
            frames = frames.astype(np.float32)
            frames = frames / 255
            framesData.append(frames)

    cap.release()
    if (framesData == []):
        # stride too big for this video
        return(-1,0)
    framesData = np.array(framesData)
    framesData = np.squeeze(framesData,axis=1)
    rval = (framesData, 1)
    return rval


def load_next_test_data_segmentation(fileName, segmentation_path, stride):

    debug = False

    # Load the video
    vid = VideoReaderOpenCV(fileName, as_float=True, grayscale=True)

    # Load the segmentations and boxes
    seg_masks = fast_seg.load_segmentations(fileName, segmentation_path)
    seg_boxes = fast_seg.segmentations_to_boxes(seg_masks)

    masks_correct = np.loadtxt("/home/trunia1/data/VideoCountingDataset/QUVACount_Segments/localization/fast_video_segment_correct.txt")
    masks_correct = masks_correct.astype(np.bool)

    # Frames in current block and segmentation masks for this block
    curr_frames = []
    curr_boxes  = []

    # List of blocks to pass to the CNN
    framesData = []

    # First frame in the video has no segmentation (no flow) so we advance one frame
    vid.next_frame()

    # Current frame index
    index = 0

    while True:

        ret, frame = vid.next_frame()
        if not ret:
            break

        if index % stride == 0:

            # Frame is catched by this stride
            # Push the frame and segmentation mask
            curr_frames.append(frame)

            box_to_add = seg_boxes[index]
            if not fast_seg.box_is_correct(box_to_add):
                # Take the entire frame if incorrect
                box_to_add = 0, 0, frame.shape[1], frame.shape[0]

            # Check if the segmentation mask is correct, if not use entire frame
            if 'LevyWolf' not in fileName:
                # Dealing with a video from QUVA-Count,
                # check whether the localization is correct
                vid_name = cortex.utils.basename(fileName)
                vid_index = int(vid_name[0:3])
                if not masks_correct[vid_index]:
                    if index == 0:
                        print("segmentation mask NOT correct for video {}: {}".format(vid_index, vid_name))
                    box_to_add = 0, 0, frame.shape[1], frame.shape[0]

            curr_boxes.append(box_to_add)

            #print("index = {}, curr_frames.length = {}, curr_seg_masks.length = {}".format(index, len(curr_frames), len(curr_boxes)))

            # Block is full, process it
            if len(curr_frames) > 20:

                # Sliding window fashion, each time remove the first from the list
                curr_frames.pop(0)
                curr_boxes.pop(0)

                block_frames = np.asarray(curr_frames)
                block_boxes  = np.asarray(curr_boxes)

                # Compute the average bounding box (other options: intersection, union)
                #box = np.mean(block_boxes, axis=0).astype(np.int32)

                # Union of Boxes
                x1_min = np.min(block_boxes[:,0])
                y1_min = np.min(block_boxes[:,1])
                x2_max = np.max(block_boxes[:,2])
                y2_max = np.max(block_boxes[:,3])
                box = np.asarray([x1_min, y1_min, x2_max, y2_max])

                # Show the frames + boxes for debugging purpose
                if debug:
                    print(block_frames.shape)
                    print(block_boxes.shape, box.shape)
                    for show_idx in range(20):
                        frame_draw = block_frames[show_idx].copy()
                        frame_draw = cv2.cvtColor(frame_draw, cv2.COLOR_GRAY2BGR)
                        fast_seg.draw_box(frame_draw, block_boxes[show_idx], color=(0,255,255), thickness=1)
                        fast_seg.draw_box(frame_draw, box, color=(0,255,255), thickness=2)
                        cv2.imshow("frame", frame_draw)
                        cv2.waitKey(0)

                # Extract the segmentation ROI from the frame block
                frames_crop = block_frames[:,box[1]:box[3],box[0]:box[2]]
                frames = np.zeros((20,50,50), np.float32)

                # Resize all the crops
                for i in range(20):
                    frames[i] = scipy.misc.imresize(frames_crop[i,:,:], size=(50,50),interp='bilinear')

                # Format the block for CNN processing
                frames  = np.reshape(frames, (1,frames.shape[0]*frames.shape[1]*frames.shape[2]))
                frames /= 255.0

                framesData.append(frames)

                #print(frames, frames.shape)

        index += 1

    # Check if stride is too big for small video
    if not framesData:
        print("framesData is empty...")
        return -1,0

    framesData = np.asarray(framesData)
    framesData = np.squeeze(framesData,axis=1)

    # Return blocks of frames and a positive status
    return framesData, 1