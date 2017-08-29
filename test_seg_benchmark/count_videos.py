import os
import glob
import pickle
import numpy as np

from rep_test_benchmark import prepare_network
from bench_classify_online import load_and_count_video

from count_dataset.quva_count import QUVACountDataset
import cortex.count.experiments

def count_entire_quva_dataset(dataset_path):

    # Prepare the CNN in Theano
    test_set_x, test_set_y, shared_test_set_y, valid_ds, classify, batchsize = \
        prepare_network()

    dataset = QUVACountDataset(dataset_path)

    # Initialize structure for storing results
    cnt_pred = np.zeros(dataset.num_examples, np.float32)
    cnt_true = np.zeros(dataset.num_examples, np.int32)

    while dataset.has_next():
        example = dataset.next_example()
        # Actual counting of the video
        cnt_pred[example.index] = load_and_count_video(example.video_path, classify, test_set_x, batchsize)
        cnt_true[example.index] = example.rep_count
        print("Video {}. True Count = {}, Predict Count = {}"
              .format(example.name, example.rep_count, cnt_pred[example.index]))

    return cnt_true, cnt_pred

if __name__ == "__main__":

    # Analyze videos
    quva_dataset = "/home/trunia1/data/VideoCountingDataset/QUVACount_Segments/"
    cnt_true, cnt_pred = count_entire_quva_dataset(quva_dataset)

    # Save results
    results_path = "/home/trunia1/experiments/2017/20170826_LevyWolf_Online/QUVACount_Segments/no_roi"
    cortex.count.experiments.write_experiment(cnt_pred, cnt_true, results_path)
