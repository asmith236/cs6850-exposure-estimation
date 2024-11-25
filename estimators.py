from collections import defaultdict
from math import lgamma, exp
import random
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

def compute_dqjd(G, beta, mu, rho):
    """
    Compute the degree-quality joint distribution P(k, theta).
    """
    joint_dist = defaultdict(float)
    for node, data in G.nodes(data=True):
        k = G.degree[node]
        theta = data["quality"]
        if k < beta:
            continue  # Ignore nodes with degree < beta
        rho_theta = rho(theta)
        gamma_factor = (
            exp(
                lgamma(k + theta) - lgamma(beta + theta)
                + lgamma(beta + theta + 2 + mu / beta)
                - lgamma(k + theta + 3 + mu / beta)
            )
        )
        p_k_theta = rho_theta * (2 + mu / beta) * gamma_factor
        joint_dist[(k, theta)] = p_k_theta

    # Normalize the distribution
    total = sum(joint_dist.values())
    for key in joint_dist:
        joint_dist[key] /= total
    return joint_dist

def compute_nnqdd(G):
    """
    Compute the nearest-neighbor quality-degree distribution P(ell, phi | k, theta).
    """
    nnqdd = defaultdict(float)
    neighbor_counts = defaultdict(float)

    for node, data in G.nodes(data=True):
        k = G.degree[node]
        theta = data["quality"]
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            ell = G.degree[neighbor]
            phi = G.nodes[neighbor]["quality"]
            nnqdd[(ell, phi, k, theta)] += 1
            neighbor_counts[(k, theta)] += 1

    # Normalize
    for key, count in nnqdd.items():
        nnqdd[key] /= neighbor_counts[(key[2], key[3])]
    return nnqdd

def compute_nqp(dqjd):
    """
    Compute the network-level quality paradox (NQP).
    """
    numerator = 0
    denominator = 0

    for (k, theta), prob in dqjd.items():
        numerator += k * theta * prob
        denominator += k * prob

    avg_quality = sum(theta * prob for (k, theta), prob in dqjd.items())
    return (numerator / denominator) - avg_quality

def nettasinghe_estimator(graph_model, n_samples):
    """
    Estimate the average exposure using the final form estimator.
    
    Parameters:
    - graph_model: An instance of the GraphModel class containing the graph.
    - n_samples: Number of samples to use for the estimation.
    
    Returns:
    - Estimated average exposure using the final form estimator.
    """
    # Precompute necessary distributions and parameters
    dqjd = compute_dqjd(graph_model.G, graph_model.beta, graph_model.mu, graph_model.rho)
    nnqdd = compute_nnqdd(graph_model.G)
    nqp = compute_nqp(dqjd)
    avg_degree = graph_model.average_degree()

    # Sample nodes uniformly
    sampled_nodes = random.sample(graph_model.G.nodes, n_samples)
    exposure_estimate = 0

    for node in sampled_nodes:
        degree = graph_model.G.degree[node]
        quality = graph_model.G.nodes[node]["quality"]
        share = graph_model.G.nodes[node]["share"]  # f(Yi)

        # Compute nearest-neighbor corrections
        neighbor_correction = sum(
            nnqdd.get((ell, phi, degree, quality), 0)
            for ell, phi, k, theta in nnqdd.keys()
            if k == degree and theta == quality
        )

        # Apply final form equation
        weight = (avg_degree / degree) * (1 + nqp) * neighbor_correction
        exposure_estimate += weight * share * quality

    # Normalize by the number of samples
    exposure_estimate /= n_samples
    return exposure_estimate
