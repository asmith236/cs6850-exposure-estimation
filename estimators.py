import random
import matplotlib.pyplot as plt
import numpy as np

def vanilla_estimator(graph_model, n_samples):
    """
    Estimate the average exposure using the vanilla estimator.
    """
    sampled_nodes = random.sample(graph_model.G.nodes, n_samples)
    exposure_estimate = np.mean([graph_model.G.nodes[node]["share"] for node in sampled_nodes])
    return exposure_estimate

def friendship_paradox_estimator(graph_model, n_samples):
    """
    Estimate the average exposure using the friendship paradox-based estimator.
    """
    edges = list(graph_model.G.edges)
    sampled_edges = random.sample(edges, n_samples)
    exposure_estimate = 0
    for u, v in sampled_edges:
        # Randomly select one end of the edge
        node = random.choice([u, v])
        degree = graph_model.G.degree[node]
        exposure_estimate += graph_model.G.nodes[node]["share"] / degree
    exposure_estimate *= graph_model.average_degree()
    exposure_estimate /= n_samples
    return exposure_estimate

def hybrid_estimator(graph_model, n_samples, alpha=0.5):
    """
    Estimate the average exposure using the hybrid estimator.

    Parameters:
    - graph_model: An instance of the GraphModel class containing the graph.
    - n_samples: Number of samples to use for the estimation.
    - alpha: The weighting parameter (0 <= alpha <= 1) that controls the balance
             between the vanilla estimator and the friendship paradox-based estimator.

    Returns:
    - Estimated average exposure using the hybrid estimator.
    """
    sampled_nodes = random.sample(graph_model.G.nodes, n_samples)  # Uniformly sample n nodes
    avg_degree = graph_model.average_degree()  # Calculate the average degree of the graph

    exposure_estimate = 0
    for node in sampled_nodes:
        degree = graph_model.G.degree[node]  # Degree of the sampled node
        share = graph_model.G.nodes[node]["share"]  # Sharing function value (f(Xi))

        # Hybrid estimator formula
        weight = alpha * (avg_degree / degree) + (1 - alpha)  # Weighted correction
        exposure_estimate += weight * share

    # Normalize by the number of samples
    exposure_estimate /= n_samples
    return exposure_estimate
