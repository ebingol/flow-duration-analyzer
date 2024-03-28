import numpy as np
from sklearn.mixture import BayesianGaussianMixture

import utility as util


def run(file_path):
    flow_ids, data_frame_all = util.load_from_file(file_path)

    for flow_id in flow_ids:

        bgm, data_frame = util.execute_bgm(data_frame_all, flow_id, "sub_bgmm")

        data_frame = util.calculate_mahalanobi_distance(data_frame, bgm)

        data_frame = execute_sub_bgm(data_frame)

        data_frame = util.calculate_anomaly_score(data_frame, bgm)

        data_frame = util.post_process(data_frame, bgm, flow_id, "sub_bgmm")

        util.visualize(data_frame, flow_id, "sub_bgmm")


def execute_sub_bgm(data_frame):
    data_frame['sub_label'] = 0
    data_frame['sub_label_mean'] = 0.00
    data_frame['mahalanobi_distance_org'] = data_frame['mahalanobi_distance']
    # for each cluster size greter than 10 execute bgmm with initial cluster size 5 and assign the labels to
    # data_frame as sub_label
    for label in data_frame['label'].unique():
        if data_frame[data_frame['label'] == label].shape[0] > 10:
            sub_bgm = BayesianGaussianMixture(max_iter=1000, random_state=42, n_components=5,
                                              weight_concentration_prior=10000, covariance_type='full',
                                              # mean_prior=mean_array, mean_precision_prior=0.001,
                                              init_params='kmeans', verbose=1, verbose_interval=10, tol=0.0001,
                                              reg_covar=1e-06, n_init=1, warm_start=False)
            data_frame_sub = data_frame[data_frame['label'] == label]
            X_sub = np.array(data_frame[data_frame['label'] == label]['mahalanobi_distance'])
            X_sub = X_sub.reshape(-1, 1)
            sub_bgm.fit(X_sub)
            sub_labels = sub_bgm.predict(X_sub)
            data_frame_sub.loc[:, 'sub_label'] = sub_labels
            data_frame_sub.loc[:, 'sub_label_mean'] = sub_bgm.means_[sub_labels]
            # start_date aynı olan kayıtların sub_label ve sub_label_mean değerlerini data_frame_sub'tan data_frame'e
            # ata
            data_frame['sub_label_mean'] = data_frame.apply(lambda x: data_frame_sub.loc[
                data_frame_sub['start_date'] == x['start_date'], 'sub_label_mean'].values[0] if data_frame_sub.loc[
                                                                                                    data_frame_sub[
                                                                                                        'start_date'] ==
                                                                                                    x[
                                                                                                        'start_date'], 'sub_label_mean'].shape[
                                                                                                    0] > 0 else x[
                'sub_label_mean'], axis=1)
            data_frame['sub_label'] = data_frame.apply(lambda x: data_frame_sub.loc[
                data_frame_sub['start_date'] == x['start_date'], 'sub_label'].values[0] if data_frame_sub.loc[
                                                                                               data_frame_sub[
                                                                                                   'start_date'] ==
                                                                                               x[
                                                                                                   'start_date'], 'sub_label'].shape[
                                                                                               0] > 0 else x[
                'sub_label'], axis=1)
    # if sub_label_mean is not zero and is not smaller mahalanobi_distance than sub_label_mean assign sub_label_mean
    # to mahalanobi_distance
    data_frame['mahalanobi_distance'] = data_frame.apply(
        lambda x: x['sub_label_mean'] if x['sub_label_mean'] != 0 else x[
            'mahalanobi_distance'], axis=1)

    return data_frame