import network as n
import estimators as e

if __name__ == "__main__":
    # init graph
    graph_dir = "./facebook"  
    G = n.init_graph(graph_dir)

    # assign node qualities
    n.assign_node_quality_prop(G)

    # calc the dqjd-inspired estimator
    e.dqjd_estimator(G)
