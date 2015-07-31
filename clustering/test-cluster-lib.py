from clusterLib.cluster import *

if __name__ == "__main__":

    video_name = "beyonce__drunk_in_love__red_couch_session_by_dan_henig_a1puW6igXcg"
    gt_nodes = load_turker_labels(video_name)
    clusters, linkage_matrix = cluster(gt_nodes)
    plot_cluster(clusters, linkage_matrix)
