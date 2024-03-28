import numpy as np
import scipy
from sklearn.cluster import KMeans
import utility as util


def execute_kmeans(data_frame_all, flow_id):
    kmeans = KMeans(n_clusters=3, random_state=42)

    x, data_frame = util.pre_process(data_frame_all, flow_id, "kmeans")

    kmeans.fit(x)

    kmeans.predict(x)

    data_frame['label'] = kmeans.predict(x)

    # Find the cluster centers
    cluster_centers = kmeans.cluster_centers_

    # Calculate the distance from each point to its assigned cluster center
    data_frame['distance'] = [np.linalg.norm(x - cluster_centers[cluster]) for x, cluster in
                              zip(x, data_frame['label'])]

    return kmeans, data_frame


def run(file_path):
    flow_ids, data_frame_all = util.load_from_file(file_path)

    for flow_id in flow_ids:
        kmeans, data_frame = execute_kmeans(data_frame_all, flow_id)

        # calculate distance from each point to its assigned cluster center
        data_frame['mahalanobi_anomaly_score'] = (scipy.stats.rankdata(data_frame['distance'])
                                                  / len(data_frame['distance']))

        data_frame = util.calculate_anomaly_score(data_frame, flow_id, "kmeans")

        util.visualize(data_frame, flow_id, "kmeans")
