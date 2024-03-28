# Import the necessary classes
import duration_anomaly_pipeline, duration_anomaly_pipeline_nearest_cluster, duration_anomaly_pipeline_sub_bgmm
import duration_anomaly_pipeline_gmm, duration_anomaly_pipeline_kmeans


def main_basic():
    # Call the run method on these instances
    duration_anomaly_pipeline.run("/Users/ezgi-lab/FlowDuration/flows2_3_4_times_in_1_hour.csv")
    duration_anomaly_pipeline.run("/Users/ezgi-lab/FlowDuration/flows_1_hour.csv")


def main_basic_nearest_cluster():
    # Call the run method on these instances
    duration_anomaly_pipeline_nearest_cluster.run("/Users/ezgi-lab/FlowDuration/flows2_3_4_times_in_1_hour.csv")
    duration_anomaly_pipeline_nearest_cluster.run("/Users/ezgi-lab/FlowDuration/flows_1_hour.csv")


def main_sub_bgm():
    # Call the run method on these instances
    duration_anomaly_pipeline_sub_bgmm.run("/Users/ezgi-lab/FlowDuration/flows2_3_4_times_in_1_hour.csv")
    duration_anomaly_pipeline_sub_bgmm.run("/Users/ezgi-lab/FlowDuration/flows_1_hour.csv")


def main_gmm():
    # Call the run method on these instances
    duration_anomaly_pipeline_gmm.run("/Users/ezgi-lab/FlowDuration/flows2_3_4_times_in_1_hour.csv")
    duration_anomaly_pipeline_gmm.run("/Users/ezgi-lab/FlowDuration/flows_1_hour.csv")


def main_kmeans():
    # Call the run method on these instances
    duration_anomaly_pipeline_kmeans.run("/Users/ezgi-lab/FlowDuration/flows2_3_4_times_in_1_hour.csv")
    duration_anomaly_pipeline_kmeans.run("/Users/ezgi-lab/FlowDuration/flows_1_hour.csv")


if __name__ == "__main__":
    main_kmeans()
