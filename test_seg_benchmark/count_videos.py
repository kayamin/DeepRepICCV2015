import os
import glob
import pickle
import numpy as np

from rep_test_benchmark import prepare_network
from bench_classify_online import load_and_count_video


def count_quva_dataset(dataset_path):

    # Prepare the CNN in Theano
    test_set_x, test_set_y, shared_test_set_y, valid_ds, classify, batchsize = \
        prepare_network()

    video_files = glob.glob(os.path.join(dataset_path, "videos", "*.mp4"))
    video_files.sort()

    num_videos = len(video_files)
    count_results = np.zeros(num_videos, np.int32)

    for i, video_file in enumerate(video_files):

        video_name, _ = os.path.splitext(os.path.basename(video_file))
        ann_file = os.path.join(dataset_path, "annotations", "count_ann_1", video_name + ".pkl")
        if os.path.exists(ann_file):
            ann = pickle.load(open(ann_file, 'rb'))
            gt_count = ann['count']
        else:
            gt_count = "unknown"

        print(gt_count)
        continue

        print("#"*60)
        print("Video {}/{} - {}".format(i, num_videos, video_name))
        print("  True Count: {}".format(gt_count))

        # Actual counting of the video
        count_results[i] = load_and_count_video(video_file, classify, test_set_x, batchsize)

        print("  Predict Count: {}".format(count_results[i]))


    return count_results

if __name__ == "__main__":

    quva_dataset = "/home/trunia1/data/VideoCountingDatasetClean/QUVACount_Segments/"
    results_file = os.path.join(quva_dataset, "results", "levywolf_online_count.npy")
    count_results = count_quva_dataset(quva_dataset)
    #np.save(results_file, count_results)
