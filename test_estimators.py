import network_graph as n
import estimators as e
import numpy as np
import matplotlib.pyplot as plt

def test_estimators(graph_model):
    """ calculates average of estimator performance at different sample sizes
    
    params:
    graph (nx.Graph): the graph used for testing
    """
    # different sample sizes to test
    sample_sizes = [500] # [100, 500, 1000, 1500, 2000, 2500]

    # number of runs per sample size
    num_runs = 1

    alpha = 0.5 # for hybrid estimator

    # placeholder for results
    results = {
        "dqjd": [],
        "vanilla": [],
        "friendship_paradox": [],
        "hybrid": [],
        "new" : []
    }

    # run experiments
    for n_samples in sample_sizes:
        dqjd_averages = []
        vanilla_averages = []
        friendship_averages = []
        hybrid_averages = []
        new_averages = []

        for _ in range(num_runs):
            dqjd_averages.append(e.dqjd_estimator(graph_model.graph, n_samples))
            vanilla_averages.append(e.vanilla_estimator(graph_model.graph, n_samples))
            friendship_averages.append(e.friendship_paradox_estimator(graph_model.graph, n_samples))
            hybrid_averages.append(e.hybrid_estimator(graph_model.graph, n_samples, alpha))
            new_averages.append(e.new_estimator(graph_model, n_samples))

        # compute average results for each estimator
        results["dqjd"].append(np.mean(dqjd_averages))
        results["vanilla"].append(np.mean(vanilla_averages))
        results["friendship_paradox"].append(np.mean(friendship_averages))
        results["hybrid"].append(np.mean(hybrid_averages))
        results["new"].append(np.mean(new_averages))

    # plt the results
    plt.figure(figsize=(10, 6))
    for key, values in results.items():
        plt.plot(sample_sizes, values, label=key.capitalize() + " Estimator")

    plt.xlabel("Sample Size (n_samples)")
    plt.ylabel("Average Estimated Exposure")
    plt.title("Comparison of Estimators vs. Sample Size")
    plt.legend()
    plt.grid(True)
    plt.savefig("estimator_performance.png")

def test_estimators_aaron(graph):
    """ calculates average of estimator performance at different sample sizes
    
    params:
    graph (nx.Graph): the graph used for testing
    """
    # different sample sizes to test
    sample_sizes = [500] # [100, 500, 1000, 1500, 2000, 2500]

    # number of runs per sample size
    num_runs = 1

    alpha = 0.5 # for hybrid estimator

    # placeholder for results
    results = {
        "dqjd": [],
        "vanilla": [],
        "friendship_paradox": [],
        "hybrid": [],
    }

    # run experiments
    for n_samples in sample_sizes:
        dqjd_averages = []
        vanilla_averages = []
        friendship_averages = []
        hybrid_averages = []

        for _ in range(num_runs):
            dqjd_averages.append(e.new_estimator(graph, n_samples))
            vanilla_averages.append(e.vanilla_estimator(graph, n_samples))
            friendship_averages.append(e.friendship_paradox_estimator(graph, n_samples))
            hybrid_averages.append(e.hybrid_estimator(graph, n_samples, alpha))

        # compute average results for each estimator
        results["dqjd"].append(np.mean(dqjd_averages))
        results["vanilla"].append(np.mean(vanilla_averages))
        results["friendship_paradox"].append(np.mean(friendship_averages))
        results["hybrid"].append(np.mean(hybrid_averages))

    # plt the results
    plt.figure(figsize=(10, 6))
    for key, values in results.items():
        plt.plot(sample_sizes, values, label=key.capitalize() + " Estimator")

    plt.xlabel("Sample Size (n_samples)")
    plt.ylabel("Average Estimated Exposure")
    plt.title("Comparison of Estimators vs. Sample Size")
    plt.legend()
    plt.grid(True)
    plt.savefig("estimator_performance.png")

# init graph
graph_dir = "./facebook"  
# graph_model = n.GraphModel(graph_dir, rho_type='uniform')  # Use uniform distribution for rho(theta)
G = n.init_graph2()
n.assign_node_quality_dist(G)

# assign node qualities
# n.assign_node_quality_prop(G)

test_estimators_aaron(G)

# analyze the effect of alpha on the hybrid estimator
# n_samples = 1000  
# print("\nanalyzing effect of alpha on hybrid estimator...")
# e.analyze_hybrid_alpha(G, n_samples)
