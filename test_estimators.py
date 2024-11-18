import network as n
import estimators as e

# init graph
graph_dir = "./facebook"  
G = n.init_graph(graph_dir)

# assign node qualities
n.assign_node_quality_prop(G)
# n.assign_node_quality_dist(G, 'normal')

n_samples = 100

# calc dqjd-inspired estimator for n samples
e.dqjd_estimator(G, n_samples)

# calc vanilla estimator for n samples
e.vanilla_estimator(G, n_samples)

# calc friendship paradox-based estimator with n samples
e.friendship_paradox_estimator(G, n_samples)

# calc hybrid estimator
alpha = 0.5
e.hybrid_estimator(G, n_samples, alpha)
