import numpy as np
import scipy
from scipy.spatial import distance
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

import utility as util


def execute_gmm(data_frame_all, flow_id):
    gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=42)

    x, data_frame = util.pre_process(data_frame_all, flow_id, "gmm")

    gmm.fit(x)

    gmm.predict(x)

    data_frame['label'] = gmm.predict(x)

    return gmm, data_frame


def run(file_path):
    flow_ids, data_frame_all = util.load_from_file(file_path)

    for flow_id in flow_ids:
        ggm, data_frame = execute_gmm(data_frame_all, flow_id)

        # preprobality density function of the data points
        data_frame['pdf'] = np.abs(ggm.score_samples(data_frame['difference'].values.reshape(-1, 1)))

        # calculate percental rank of pdf values
        data_frame['mahalanobi_anomaly_score'] = scipy.stats.rankdata(data_frame['pdf']) / len(data_frame['pdf'])
        # scale the mahalanobi_anomaly_score to 100
        data_frame = util.calculate_anomaly_score(data_frame, flow_id, method="gmm")

        util.visualize(data_frame, flow_id, "gmm")
