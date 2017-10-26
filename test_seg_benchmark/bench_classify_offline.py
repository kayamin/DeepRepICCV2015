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


def test_benchmark_offline(classify, test_set_x, batch_size):
	
	strides = (2,5,8)
	vid_root = '../data/YT_seg/'
	gt_counts = pickle.load( open( "vidGtData.p", "rb" ) )	

	gt = numpy.tile(gt_counts, (len(strides), 1))
	gt = gt.T  
	gt1 = gt[:,1]
	tests_num = gt_counts.shape[0]	
	countArr = numpy.zeros(shape=(tests_num, len(strides)))
	entropy = numpy.zeros(shape=(tests_num, len(strides)))
	num_entropy = numpy.zeros(shape=(tests_num, len(strides)))
	
	currStride = -1
	for nStride in (strides):
		currStride += 1

		for nTestSet in range(tests_num):

			print 'stride: %i, vid: %i' % (nStride, nTestSet)
			fileName = vid_root+'YT_seg_'+str(nTestSet)+'.avi'
			mydatasets = load_next_test_data_simple_roi(fileName, nStride)

			ns_test_set_x, valid_ds = mydatasets
			if (valid_ds == 0):  # file not axists
				continue

			test_set_x.set_value(ns_test_set_x, borrow=True)
			n_samples = ns_test_set_x.shape[0]

			out_list = [classify(i) for i in xrange(n_samples)]

			frame_counter = 0
			rep_counter = 0			
			curr_entropy = 0
			ent_cnt = 0

			for batch_num in range(len(out_list)):

				output_label_batch , pYgivenX = out_list[batch_num]

				# Removed index in following line
				output_label = output_label_batch[0] + 3 # moving from label to cycle length
				pYgivenX[pYgivenX==0] = numpy.float32(1e-30) # hack to output valid entropy

				# calc entropy
				curr_entropy = curr_entropy - (pYgivenX*numpy.log(pYgivenX)).sum()
				ent_cnt= ent_cnt + 1
				# integrate counting
				if (batch_num==0):
					rep_counter = 20 / (output_label)
					frame_counter = 20 % (output_label)
				else:
					frame_counter += 1	
					if (frame_counter >= output_label):
						rep_counter += 1;
						frame_counter = 0;
		
			countArr[nTestSet, currStride] = rep_counter
			entropy[nTestSet, currStride] = curr_entropy
			num_entropy[nTestSet, currStride] = ent_cnt
	
	
	absdiff_o = abs(countArr-gt)
	min_err_cnt_o = absdiff_o.min(axis=1)    
	min_err_perc_o = min_err_cnt_o/gt[:,1]	
	err_perc_o = numpy.average(min_err_perc_o)*100
	print 'alpha = 1: precentage error:    %.2f%%' % (err_perc_o)


	print '---------'
	med = numpy.median(countArr,axis=1)
	medif = numpy.average(abs(med-gt1)/gt1)*100	
	print 'median stride: precentage error:    %.2f%%' % medif
	
		
	xx = entropy/num_entropy	
	chosen_stride = numpy.nanargmin(xx,axis=1)
	m = numpy.arange(chosen_stride.shape[0])*len(strides)
	m = m + chosen_stride
	flt = countArr.flatten()
	ent_chosen_cnt = flt[m]
	
	entdif = numpy.average(abs(ent_chosen_cnt-gt1)/gt1)*100
	print 'enropy stride: precentage error:    %.2f%%' % entdif
	
	print 'offline entropy counting done'
		