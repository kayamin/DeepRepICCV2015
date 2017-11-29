import os
import glob
import pickle
import numpy as np

from rep_test_benchmark import prepare_network
from bench_classify_online import load_and_count_video

import count_dataset
import cortex.count

def count_quvacount(dataset, localization_method, segmentation_path):
    '''

    Count entire QUVACount dataset using the online method.
    When using the fullsystem this should return approximately these results (20170925):

      Num Examples:   100
      Mean Abs. Err.: 45.668 +- 58.710
      Mean Sq. Err.:  86.390
      Accuracy:       0.190
      OBOA:           0.440
      Within-10%:     0.280
      Frac. Zero:     0.000

    :param dataset_path:
    :return:
    '''

    # Prepare the CNN in Theano
    test_set_x, test_set_y, shared_test_set_y, valid_ds, classify, batchsize = \
        prepare_network()


    # Initialize structure for storing results
    cnt_pred = np.zeros(dataset.num_examples, np.float32)
    cnt_true = np.zeros(dataset.num_examples, np.int32)

    while dataset.has_next():
        example = dataset.next_example()

        # Actual counting of the video
        cnt_pred[example.index] = load_and_count_video(
            example.video_path, classify, test_set_x, batchsize,
            localization_method, segmentation_path)

        cnt_true[example.index] = example.rep_count
        print("Video {}. True Count = {}, Predict Count = {}"
              .format(example.name, example.rep_count, cnt_pred[example.index]))

    return cnt_true, cnt_pred

def count_quvacount_accelerate(dataset_path, vid_accelate_path):
    '''

    Same as above but on a subset of videos in which the last half is accelerated

    Acceleration 2.0x results (2017-09-25):

          Num Examples:   36
          Mean Abs. Err.: 29.407 +- 21.347
          Mean Sq. Err.:  132.222
          Accuracy:       0.028
          OBOA:           0.417
          Within-10%:     0.111
          Frac. Zero:     0.000

    Acceleration 0.5x results (2017-09-25):

          Num Examples:   36
          Mean Abs. Err.: 92.063 +- 91.985
          Mean Sq. Err.:  268.083
          Accuracy:       0.028
          OBOA:           0.250
          Within-10%:     0.139
          Frac. Zero:     0.000

    :param dataset_path:
    :param vid_accelate_path:
    :return:
    '''

    # Prepare the CNN in Theano
    test_set_x, test_set_y, shared_test_set_y, valid_ds, classify, batchsize = \
        prepare_network()


    dataset = QUVACountDataset(dataset_path)

    # Initialize structure for storing results
    cnt_pred = []
    cnt_true = []

    while dataset.has_next():
        example = dataset.next_example()
        vid_accelerate = os.path.join(vid_accelate_path, example.name + ".avi")
        print(vid_accelerate)

        # Check if there is an accelerated version of it
        if not os.path.exists(vid_accelerate):
            continue

        # Actual counting of the video
        cnt_pred_curr = load_and_count_video(vid_accelerate, classify, test_set_x, batchsize)[0]

        cnt_pred.append(cnt_pred_curr)
        cnt_true.append(example.rep_count)

        print("  True Count = {}, Predict Count = {}".format(example.rep_count, cnt_pred[-1]))

    cnt_pred = np.asarray(cnt_pred, np.int32)
    cnt_true = np.asarray(cnt_true, np.int32)

    return cnt_true, cnt_pred


if __name__ == "__main__":

    # Count videos (main experiment)
    dataset = count_dataset.init_quva_dataset(speed=1.0)

    localization_method = 'simple'
    segmentation_path = None

    #segmentation_path = "/home/trunia1/data/VideoCountingDataset/QUVACount_Segments/localization/FastVideoSegment/"

    # Count the entire dataset
    cnt_true, cnt_pred = count_quvacount(dataset, localization_method, segmentation_path)

    # Count videos (accelerate subset)
    #accelate_video_path = "/home/trunia1/data/VideoCountingDataset/QUVACount_Segments/videos_acceleration/accelerate_0.5/"
    #cnt_true, cnt_pred = count_quvacount_accelerate(dataset, accelate_video_path)

    # Save results
    results_path = "/home/trunia1/experiments/2017/20170925_LevyWolf_FINAL/online/QUVACount_Segments/{}/".format(localization_method)
    cortex.count.write_experiment(cnt_pred, dataset, results_path)