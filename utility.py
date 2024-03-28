import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import distance
from scipy.stats import rankdata
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
import seaborn as sns


def scale_to_percentage(data_frame):
    # rank the mahalanobi_distance
    data_frame['rank'] = rankdata(data_frame['mahalanobi_distance'])
    # calculate the percentage of mahalanobi_distance
    data_frame['mahalanobi_anomaly_score'] = (data_frame['rank'] / data_frame.shape[0]) * 100
    return data_frame['mahalanobi_anomaly_score']


# Function to calculate Mahalanobis distance to the nearest cluster
def mahalanobis_distance_to_its_cluster(x, mean, cov):
    point = np.array([x])

    inv_cov = np.linalg.inv(cov - 1e-6 * np.eye(cov.shape[0]))  # Add small value to diagonal to avoid singularity
    distance_ = distance.mahalanobis(point, mean, inv_cov)

    return distance_, cov  # Return distance and cluster index


def post_process(data_frame, bgm, flow_id, method_name):
    # create a data_frame for components of bgm
    # columns: mean, variance, weight_concentration_prior, weights
    # mean: mean of each component
    # variance: standard deviation of each component
    # weight_concentration_prior: weight concentration prior of each component
    # weights: weights of each component
    data_frame_components = pd.DataFrame(bgm.means_, columns=['mean'])
    data_frame_components['weight_concentration_prior'] = bgm.weight_concentration_prior_
    # calculate the variance of each component
    # data_frame_components['variance'] = np.sqrt(bgm.covariances_)
    data_frame_components['weights'] = bgm.weights_
    # sort bgm components by weights
    data_frame_components = data_frame_components.sort_values(by=['weights'], ascending=False).reset_index(drop=True)
    # save  the data_frame to /Users/ezgi-lab/FlowDuration/flow_id_gmm.csv
    data_frame.to_csv(
        '/Users/ezgi-lab/FlowDuration/BGMM/' + method_name + '/FlowId_' + str(flow_id) + '/flow_id_bgmm' + str(
            flow_id) + '.csv',
        index=False)
    # save the data_frame_components to /Users/ezgi-lab/FlowDuration/flow_id_components.csv
    data_frame_components.to_csv(
        '/Users/ezgi-lab/FlowDuration/BGMM/' + method_name + '/FlowId_' + str(flow_id) + '/flow_id_components' + str(
            flow_id) + '.csv',
        index=False)

    return data_frame


def visualize(data_frame, flow_id, method_name):
    #   'Difference between start and end dates of flows')
    setFig = plt.figure(figsize=(20, 10))
    sns.lineplot(x='start_date', y='difference', data=data_frame, color='grey')
    df_mahalanobi_10 = data_frame[data_frame['mahalanobi_anomaly_10'] == True]
    df_mahalanobi_5 = data_frame[data_frame['mahalanobi_anomaly_5'] == True]
    sns.stripplot(data=df_mahalanobi_10, x="start_date", y="difference", color='purple', s=10)
    sns.stripplot(data=df_mahalanobi_5, x="start_date", y="difference", color='red', s=10)
    # save the plot to  /Users/ezgi-lab/FlowDuration/flow_id.png
    plt.savefig(
        '/Users/ezgi-lab/FlowDuration/BGMM/' + method_name + '/FlowId_' + str(flow_id) + '/anomaly_bgmm.png')
    # close the plot
    plt.close()
    # clear the plot
    plt.clf()
    # clear the plot
    plt.cla()
    # clear the plot
    plt.close()
    # clear the plot
    plt.close('all')


def calculate_anomaly_score(data_frame, bgm):
    weights = bgm.weights_

    # label of maximum weight
    max_label = np.argmax(weights)

    # calculate minimum mahalanobi_distance
    min_distance = data_frame['mahalanobi_distance'].min()

    data_frame['mahalanobi_anomaly_score'] = 0.

    data_frame['mahalanobi_anomaly_score'] = scale_to_percentage(data_frame)

    # values smaller than 90 percentil of data but anomaly score greater than 90 set to 0
    data_frame['mahalanobi_distance'] = data_frame.apply(
        lambda x: min_distance if x['difference'] < np.percentile(data_frame['difference'], 90) and x[
            'mahalanobi_anomaly_score'] > 90 else x['mahalanobi_distance'], axis=1)

    data_frame['mahalanobi_anomaly_score'] = scale_to_percentage(data_frame)

    data_frame['mahalanobi_anomaly'] = data_frame['mahalanobi_anomaly_score'] > 90
    # label max_label ise mahalanobi_anomaly false
    data_frame['mahalanobi_anomaly'] = data_frame.apply(
        lambda x: False if x['label'] == max_label else x['mahalanobi_anomaly'], axis=1)
    # difference percentile 90'dan küçük ise mahalanobi_anomaly false
    data_frame['mahalanobi_anomaly'] = data_frame.apply(
        lambda x: False if x['difference'] < np.percentile(data_frame['difference'], 90)
        else x['mahalanobi_anomaly'], axis=1)
    # mahalanobi_anomaly_score 10 dan küçük ama mahalanobi_anomaly_score 5'den büyük ise  ve mahalanobi_anomaly true
    # ise mahalanobi_anomaly_10 true
    data_frame['mahalanobi_anomaly_10'] = data_frame.apply(lambda x:
                                                           True if 90 < x['mahalanobi_anomaly_score'] < 95 and x[
                                                               'mahalanobi_anomaly'] else False, axis=1)
    # mahalanobi_anomaly_score 5'den küçük ise ve ve mahalanobi_anomaly true  mahalanobi_anomaly_5 true
    data_frame['mahalanobi_anomaly_5'] = data_frame.apply(
        lambda x: True if x['mahalanobi_anomaly_score'] > 95 and x['mahalanobi_anomaly'] else False, axis=1)

    return data_frame


def calculate_mahalanobi_distance(data_frame, bgm):
    # label with min size
    min_label = data_frame.groupby('label').size().idxmin()

    data_frame['mahalanobi_distance'] = \
        data_frame.apply(lambda x: mahalanobis_distance_to_its_cluster(x['difference'],
                                                                       bgm.means_[
                                                                           x['label']]
                                                                       * bgm.weights_[
                                                                           x['label']]
                                                                       ,
                                                                       bgm.covariances_[
                                                                           x['label']]
                                                                       * bgm.weights_[
                                                                           x['label']])[
            0], axis=1)
    data_frame['variance'] = data_frame.apply(lambda x: bgm.covariances_[x['label']][0], axis=1)
    # min_label clusterdaki en yüksek distance'ı bul
    max_distance = data_frame[data_frame['label'] == min_label]['mahalanobi_distance'].max()
    # min_label cluster size
    min_label_size = data_frame[data_frame['label'] == min_label].shape[0]
    # if min_label_size < 3 ise mahalanobi_distance max_distance yap
    data_frame['mahalanobi_distance'] = data_frame.apply(lambda x: max_distance if x['label'] == min_label
                                                                                   and min_label_size < 3 else x[
        'mahalanobi_distance'], axis=1)

    return data_frame


def load_from_file(file_path):
    data_frame_all = pd.read_csv(file_path, delimiter=',')

    return data_frame_all['flow_id'].unique(), data_frame_all


def pre_process(data_frame_all, flow_id, method_name="basic"):
    data_frame = data_frame_all[data_frame_all['flow_id'] == flow_id]
    if not os.path.exists('/Users/ezgi-lab/FlowDuration/BGMM/' + method_name + '/FlowId_' + str(flow_id)):
        os.makedirs('/Users/ezgi-lab/FlowDuration/BGMM/' + method_name + '/FlowId_' + str(flow_id))
    data_frame = data_frame.sort_values(by=['start_date'], ascending=True).reset_index(drop=True)
    data_frame['original_dif'] = data_frame['difference']
    # Apply standardization for 1-D data_frame['difference']
    scaler = StandardScaler()
    data_frame['difference'] = scaler.fit_transform(data_frame['difference'].values.reshape(-1, 1))
    # create a numpy array from difference column in data_frame
    X = np.array(data_frame['difference'])
    # reshape X
    X = X.reshape(-1, 1)
    return X, data_frame


def execute_bgm(data_frame_all, flow_id, method_name="basic"):
    x, data_frame = pre_process(data_frame_all, flow_id, method_name)
    # create a BayesianGaussianMixture object
    bgm = BayesianGaussianMixture(max_iter=1000, random_state=42, n_components=10,
                                  weight_concentration_prior=0.1, covariance_type='full',
                                  # mean_prior=mean_array, mean_precision_prior=0.001,
                                  init_params='kmeans', verbose=1, verbose_interval=10, tol=0.0001,
                                  reg_covar=1e-06, n_init=10, warm_start=False)
    # fit the model to X
    bgm.fit(x)
    # predict the labels of X
    labels = bgm.predict(x)
    # create a new column in data_frame with name 'label' and assign labels to it
    data_frame['label'] = labels
    return bgm, data_frame


def calculate_anomaly_score(data_frame, flow_id, method="kmeans"):
    # scale the mahalanobi_anomaly_score to 100
    data_frame['mahalanobi_anomaly_score'] = data_frame['mahalanobi_anomaly_score'] * 100
    data_frame['mahalanobi_anomaly'] = data_frame['mahalanobi_anomaly_score'] > 90
    # difference percentile 90'dan küçük ise mahalanobi_anomaly false
    data_frame['mahalanobi_anomaly'] = data_frame.apply(
        lambda x: False if x['difference'] < np.percentile(data_frame['difference'], 90)
        else x['mahalanobi_anomaly'], axis=1)
    # mahalanobi_anomaly_score 10 dan küçük ama mahalanobi_anomaly_score 5'den büyük ise  ve mahalanobi_anomaly true
    # ise mahalanobi_anomaly_10 true
    data_frame['mahalanobi_anomaly_10'] = data_frame.apply(lambda x:
                                                           True if 90 < x['mahalanobi_anomaly_score'] < 95 and x[
                                                               'mahalanobi_anomaly'] else False, axis=1)
    # mahalanobi_anomaly_score 5'den küçük ise ve ve mahalanobi_anomaly true  mahalanobi_anomaly_5 true
    data_frame['mahalanobi_anomaly_5'] = data_frame.apply(
        lambda x: True if x['mahalanobi_anomaly_score'] > 95 and x['mahalanobi_anomaly'] else False, axis=1)
    # save  the data_frame to /Users/ezgi-lab/FlowDuration/flow_id_gmm.csv
    data_frame.to_csv(
        '/Users/ezgi-lab/FlowDuration/BGMM/' + method + '/FlowId_' + str(flow_id) + '/flow_id_' + str(
            flow_id) + '.csv',
        index=False)

    return data_frame
