

import utility as util


def run(file_path):

    flow_ids, data_frame_all = util.load_from_file(file_path)

    for flow_id in flow_ids:

        bgm, data_frame = util.execute_bgm(data_frame_all, flow_id)

        data_frame = util.calculate_mahalanobi_distance(data_frame, bgm)

        data_frame = util.calculate_anomaly_score(data_frame, bgm)

        data_frame = util.post_process(data_frame, bgm, flow_id, "basic")

        util.visualize(data_frame, flow_id, "basic")
