# Description: This file contains the pipeline for the duration anomaly detection using the nearest cluster method.
import numpy as np

import utility as util


# find nearest cluster to the point x
def find_nearest_cluster(x, means, covariances):
    distances = [util.mahalanobis_distance_to_its_cluster(x, mean, cov)[0] for mean, cov in zip(means, covariances)]
    return np.argmin(distances)


def run(file_path):
    flow_ids, data_frame_all = util.load_from_file(file_path)

    for flow_id in flow_ids:
        bgm, data_frame = util.execute_bgm(data_frame_all, flow_id, "basic_nearest_cluster")

        data_frame = util.calculate_mahalanobi_distance(data_frame, bgm)

        # find for each data point the nearest cluster
        data_frame['nearest_cluster'] = data_frame.apply(
            lambda x: find_nearest_cluster(x['difference'], bgm.means_,
                                                bgm.covariances_), axis=1)

        # assign label to origin_label
        data_frame['origin_label'] = data_frame['label']
        # assign the nearest cluster to label
        data_frame['label'] = data_frame['nearest_cluster']

        data_frame = util.calculate_anomaly_score(data_frame, bgm)

        data_frame = util.post_process(data_frame, bgm, flow_id, "basic_nearest_cluster")

        util.visualize(data_frame, flow_id, "basic_nearest_cluster")
