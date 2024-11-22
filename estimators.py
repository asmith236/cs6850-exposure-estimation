
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import numpy as np

# TODO identify why dqjd estimator is always approx 1
import numpy as np
import networkx as nx

import random
import numpy as np
import networkx as nx


# CURRENT WORKING
# def dqjd_estimator(graph, n_samples):
#     """
#     Implements the exposure formula using NQP and NFP with sampled nodes.

#     This function calculates the exposure of a sampled subset of nodes in the graph
#     based on the degree-quality joint distribution (NQP) and network-level degree variance (NFP).
#     """
#     # Step 1: Hybrid sampling of `n_samples` nodes
#     half_samples = n_samples // 2
#     uniform_sampled_nodes = random.sample(list(graph.nodes()), half_samples)
#     degree_weighted_sampled_nodes = random.choices(
#         list(graph.nodes()),
#         weights=[graph.degree[node] for node in graph.nodes()],
#         k=n_samples - half_samples
#     )
#     sampled_nodes = uniform_sampled_nodes + degree_weighted_sampled_nodes

#     # Step 2: Calculate global metrics with normalization
#     qualities = np.array([graph.nodes[node].get('quality', 0) for node in graph.nodes()])
#     degrees = np.array([graph.degree[node] for node in graph.nodes()])
#     max_quality = np.max(qualities) if np.max(qualities) > 0 else 1
#     normalized_qualities = qualities / max_quality
#     max_degree = np.max(degrees) if np.max(degrees) > 0 else 1
#     normalized_degrees = degrees / max_degree

#     mean_quality = np.mean(normalized_qualities)
#     mean_degree = np.mean(normalized_degrees)
#     degree_variance = np.var(normalized_degrees)

#     # Compute NQP
#     total_neighbors_quality = sum(normalized_degrees[i] * normalized_qualities[i] for i in range(len(degrees)))
#     total_neighbors = sum(normalized_degrees)
#     nqp = (total_neighbors_quality / total_neighbors) / mean_quality - 1
#     nqp = max(0, nqp)  # Ensure non-negative NQP

#     # Compute NFP
#     nfp = degree_variance / mean_degree
#     nfp = max(0, min(nfp, 1))  # Restrict NFP to [0, 1]

#     # Step 3: Calculate exposure for the sampled nodes
#     exposures = []
#     for node in sampled_nodes:
#         neighbors = list(graph.neighbors(node))
#         if not neighbors:
#             exposures.append(graph.nodes[node].get('quality', 0))
#             continue

#         # Local component
#         local_neighbors_quality = sum(
#             graph.degree[neighbor] * graph.nodes[neighbor].get('quality', 0) for neighbor in neighbors
#         )
#         local_exposure = local_neighbors_quality / (len(neighbors) * max_quality)  # Normalized by max quality

#         # Incorporate global adjustments with scaling
#         node_degree = graph.degree[node] / max_degree
#         nqp_adjustment = 0.1 * nqp * (node_degree / mean_degree)
#         nfp_adjustment = 0.1 * nfp * (degree_variance / mean_degree)
#         global_adjustment = 1 + nqp_adjustment + nfp_adjustment
#         global_adjustment = max(0, min(global_adjustment, 1.5))  # Restrict to [0, 1.5]

#         exposure = local_exposure * global_adjustment
#         exposures.append(exposure)

#     # Step 4: Return the average exposure across the sampled nodes
#     estimated_average_exposure = np.mean(exposures)
#     print(f"estimated average exposure using dqjd estimator = {estimated_average_exposure:.4f}")
#     return estimated_average_exposure



# def dqjd_estimator(graph, n_samples):
#     """
#     Implements the exposure formula using NQP and NFP with sampled nodes.

#     This function calculates the exposure of a sampled subset of nodes in the graph
#     based on the degree-quality joint distribution (NQP) and network-level degree variance (NFP).
#     """
#     # Step 1: Hybrid sampling of `n_samples` nodes
#     half_samples = n_samples // 2
#     uniform_sampled_nodes = random.sample(list(graph.nodes()), half_samples)
#     degree_weighted_sampled_nodes = random.choices(
#         list(graph.nodes()),
#         weights=[graph.degree[node] for node in graph.nodes()],
#         k=n_samples - half_samples
#     )
#     sampled_nodes = uniform_sampled_nodes + degree_weighted_sampled_nodes

#     # Step 2: Calculate global metrics with normalization
#     qualities = np.array([graph.nodes[node].get('quality', 0) for node in graph.nodes()])
#     degrees = np.array([graph.degree[node] for node in graph.nodes()])
#     max_quality = np.max(qualities) if np.max(qualities) > 0 else 1
#     normalized_qualities = qualities / max_quality
#     max_degree = np.max(degrees) if np.max(degrees) > 0 else 1
#     normalized_degrees = degrees / max_degree

#     mean_quality = np.mean(normalized_qualities)
#     mean_degree = np.mean(normalized_degrees)
#     degree_variance = np.var(normalized_degrees)

#     # Compute NQP
#     total_neighbors_quality = sum(normalized_degrees[i] * normalized_qualities[i] for i in range(len(degrees)))
#     total_neighbors = sum(normalized_degrees)
#     nqp = (total_neighbors_quality / total_neighbors) - mean_quality
#     nqp = max(min(nqp, 1), -1)  # Clip to [-1, 1]

#     # Compute NFP
#     mean_squared_degree = np.mean(normalized_degrees ** 2)
#     nfp = (mean_squared_degree - mean_degree ** 2) / mean_degree
#     nfp = max(min(nfp, 1), -1)  # Clip to [-1, 1]

#     # Step 3: Calculate exposure for the sampled nodes
#     exposures = []
#     for node in sampled_nodes:
#         neighbors = list(graph.neighbors(node))
#         if not neighbors:
#             exposures.append(graph.nodes[node].get('quality', 0))
#             continue

#         # Local component
#         local_neighbors_quality = sum(
#             graph.degree[neighbor] * graph.nodes[neighbor].get('quality', 0) for neighbor in neighbors
#         )
#         local_exposure = local_neighbors_quality / (len(neighbors) * max_quality + 1e-5)  # Normalized

#         # Incorporate global adjustments with scaling
#         node_degree = graph.degree[node] / max_degree
#         nqp_adjustment = 0.1 * nqp * (node_degree / (mean_degree + 1e-5))
#         nfp_adjustment = 0.1 * nfp * (degree_variance / (mean_degree + 1e-5))
#         global_adjustment = max((1 + nqp_adjustment) * (1 + nfp_adjustment), 0.1)  # Ensure non-negative

#         exposure = local_exposure * global_adjustment
#         exposures.append(exposure)

#         # Debugging output
#         # print(f"Node: {node}, Local Exposure: {local_exposure:.4f}, NQP Adjustment: {nqp_adjustment:.4f}, NFP Adjustment: {nfp_adjustment:.4f}, Global Adjustment: {global_adjustment:.4f}")

#     # Step 4: Return the average exposure across the sampled nodes
#     estimated_average_exposure = np.mean(exposures)
#     print(f"estimated average exposure using dqjd estimator = {estimated_average_exposure:.4f}")
#     return estimated_average_exposure


def dqjd_estimator(graph, n_samples):
    """
    Implements the exposure formula using NQP and NFP with sampled nodes.

    This function calculates the exposure of a sampled subset of nodes in the graph
    based on the degree-quality joint distribution (NQP) and network-level degree variance (NFP).
    """
    # Step 1: Sample nodes uniformly at random
    sampled_nodes = random.sample(list(graph.nodes()), n_samples)

    # Step 2: Calculate global metrics (based on the entire graph)
    qualities = np.array([graph.nodes[node].get('quality', 0) for node in graph.nodes()])
    degrees = np.array([graph.degree[node] for node in graph.nodes()])
    mu = np.mean(qualities)
    var_k = np.var(degrees)
    

    # Step 3: Calculate exposure for the sampled nodes
    exposures = []
    nqp_num = 0
    nqp_den = 0
    nfp_ksquare = 0
    nfp_k = 0
    for node in graph.nodes():
        k = graph.degree[node]
        theta = graph.nodes[node].get('quality', 0)

        pk_theta = k ** (-3 - mu/k)
        nqp_num += pk_theta * k * theta
        nqp_den += pk_theta * k
        nfp_ksquare += k ** 2
        nfp_k += k

    nqp = nqp_num / nqp_den - mu
    avg_ksquare = nfp_ksquare / n_samples
    avg_k = nfp_k / n_samples
    nfp = (avg_ksquare - (avg_k ** 2)) / avg_k

    for node in sampled_nodes:
        weighted_sum = 0
        k = graph.degree[node]
        theta = graph.nodes[node].get('quality', 0)
        adjustment = (1 + nqp * (k / avg_k)) * (1 + nfp * (var_k / avg_k))
        
        neighbors = list(graph.neighbors(node))
        for neighbor in neighbors:
            weighted_sum += graph.degree[neighbor] * graph.nodes[neighbor].get('quality', 0)

        if not neighbors:
            local_exposure = 0
        else:
            local_exposure = (weighted_sum / len(neighbors)) * adjustment
    
        exposures.append(local_exposure)

    # Step 4: Return the average exposure across the sampled nodes
    estimated_average_exposure = np.mean(exposures)
    print(f"estimated average exposure using dqjd estimator = {estimated_average_exposure:.4f}")
    return estimated_average_exposure





def vanilla_estimator(graph, n):
    """ implements the vanilla estimator method
    
    samples n random nodes from the graph and estimates the average exposure
    
    params:
    graph (nx.Graph): the graph for which the estimation is performed
    n (int): the number of nodes to sample.
    
    returns:
    float: estimated average exposure of the sampled nodes
    """
    sampled_nodes = random.sample(list(graph.nodes()), n)
    total_exposure = 0

    def exposure_function(node):
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            return 0
        exposed_neighbors = sum(1 for neighbor in neighbors if graph.nodes[neighbor].get('quality', 0) > 0.5)
        normalized_degree = graph.degree[node] / max(dict(graph.degree()).values())
        return min(1, 1 / (1 + 2 ** -(0.2 * normalized_degree))) * (exposed_neighbors / len(neighbors))

    for node in sampled_nodes:
        total_exposure += exposure_function(node)

    estimated_average_exposure = total_exposure / n
    print(f"estimated average exposure using vanilla estimator = {estimated_average_exposure:.4f}")
    return estimated_average_exposure


def friendship_paradox_estimator(graph, n):
    """ implements the friendship paradox-based estimator method
    
    samples n random friends (neighbors) from the graph and estimates the average exposure
    
    params:
    graph (nx.Graph): the graph for which the estimation is performed
    n (int): the number of friends to sample
    
    returns:
    float: estimated average exposure of the sampled friends
    """
    sampled_edges = random.sample(list(graph.edges()), n)
    sampled_nodes = [random.choice(edge) for edge in sampled_edges]
    total_exposure = 0
    average_degree = sum(dict(graph.degree()).values()) / graph.number_of_nodes()

    def exposure_function(node):
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            return 0
        exposed_neighbors = sum(1 for neighbor in neighbors if graph.nodes[neighbor].get('quality', 0) > 0.5)
        normalized_degree = graph.degree[node] / max(dict(graph.degree()).values())
        return min(1, 1 / (1 + 2 ** -(0.2 * normalized_degree))) * (exposed_neighbors / len(neighbors))

    for node in sampled_nodes:
        degree = graph.degree[node]
        total_exposure += (exposure_function(node) / degree)

    estimated_average_exposure = (average_degree / n) * total_exposure
    print(f"estimated average exposure using friendship paradox-based estimator = {estimated_average_exposure:.4f}")
    return estimated_average_exposure


def hybrid_estimator(graph, n, alpha):
    """
    computes the hybrid estimator
    params:
    graph (nx.Graph): the graph for which the estimation is performed
    n (int): the number of nodes to sample
    alpha (float): the degree of weighting, ranges from 0 to 1
    returns:
    float: the estimated value based on the hybrid estimator
    """
    sampled_nodes = random.sample(list(graph.nodes()), n)
    avg_degree = sum(dict(graph.degree()).values()) / graph.number_of_nodes()
    total_weighted_exposure = 0

    def exposure_function(node):
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            return 0
        exposed_neighbors = sum(1 for neighbor in neighbors if graph.nodes[neighbor].get('quality', 0) > 0.5)
        normalized_degree = graph.degree[node] / max(dict(graph.degree()).values())
        return min(1, 1 / (1 + 2 ** -(0.2 * normalized_degree))) * (exposed_neighbors / len(neighbors))

    for node in sampled_nodes:
        degree = graph.degree[node]
        weight = min(1, alpha * (avg_degree / degree) + (1 - alpha)) if degree > 0 else 1 - alpha
        exposure = exposure_function(node)
        total_weighted_exposure += weight * exposure

    estimated_average_exposure = (1 / n) * total_weighted_exposure
    print(f"estimated average exposure using hybrid estimator = {estimated_average_exposure:.4f}")
    return estimated_average_exposure


def analyze_hybrid_alpha(graph, n):
    """
    analyzes the effect of alpha on the hybrid estimator
    
    params:
    graph (nx.Graph): the input graph
    n (int): the number of nodes to sample
    """
    alpha_values = [i / 10 for i in range(11)]  # 0.0 to 1.0 in steps of 0.1
    results = []

    for alpha in alpha_values:
        print(f"testing hybrid estimator with alpha = {alpha:.1f}")
        estimated_exposure = hybrid_estimator(graph, n, alpha)
        results.append((alpha, estimated_exposure))
    
    # print results
    for alpha, exposure in results:
        print(f"alpha = {alpha:.1f}, estimated exposure = {exposure:.4f}")
    
    # plot the results (requires matplotlib)
    alphas, exposures = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, exposures, marker='o')
    plt.xlabel('Alpha')
    plt.ylabel('Estimated Exposure')
    plt.title('Effect of Alpha on Hybrid Estimator')
    plt.grid(True)
    plt.savefig("hybrid_alpha.png")