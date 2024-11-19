
from collections import defaultdict
import random

# TODO identify why dqjd estimator is always approx 1
def dqjd_estimator(graph, n):
    """ Implements the dqjd-inspired estimator with improved normalization and smoothing. """

    sampled_nodes = random.sample(list(graph.nodes()), n)
    degree_counts = defaultdict(int)
    quality_counts = defaultdict(int)
    joint_counts = defaultdict(lambda: defaultdict(int))

    # Compute normalization factors
    max_degree = max(dict(graph.degree()).values())
    max_quality = max(graph.nodes[node]['quality'] for node in graph.nodes())

    for node in sampled_nodes:
        degree = graph.degree[node]
        quality = graph.nodes[node].get('quality', 0)

        degree_counts[degree] += 1
        quality_counts[quality] += 1
        joint_counts[degree][quality] += 1

    total_nodes = len(sampled_nodes)

    p_k_theta = {}
    for degree in joint_counts:
        p_k_theta[degree] = {}
        for quality in joint_counts[degree]:
            p_k_theta[degree][quality] = joint_counts[degree][quality] / total_nodes

    def f_k_theta(degree, quality):
        normalized_degree = degree / max_degree if max_degree > 0 else 0
        normalized_quality = quality / max_quality if max_quality > 0 else 0
        return min(1, 1 / (1 + 2 ** -(0.5 * normalized_degree + 0.5 * normalized_quality)))

    estimated_exposure = 0
    for degree in p_k_theta:
        for quality in p_k_theta[degree]:
            estimated_exposure += p_k_theta[degree][quality] * f_k_theta(degree, quality)

    print(f"Estimated fraction of exposed nodes using dqjd-inspired estimator = {estimated_exposure:.4f}")
    return estimated_exposure


def vanilla_estimator(graph, n):
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
    print(f"Estimated average exposure using vanilla estimator = {estimated_average_exposure:.4f}")
    return estimated_average_exposure


def friendship_paradox_estimator(graph, n):
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
    print(f"Estimated average exposure using friendship paradox-based estimator = {estimated_average_exposure:.4f}")
    return estimated_average_exposure


def hybrid_estimator(graph, n, alpha):
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
    print(f"Estimated average exposure using hybrid estimator = {estimated_average_exposure:.4f}")
    return estimated_average_exposure


def analyze_hybrid_alpha(graph, n):
    """
    Analyzes the effect of alpha on the hybrid estimator.
    
    Parameters:
    graph (nx.Graph): The input graph.
    n (int): The number of nodes to sample.
    """
    alpha_values = [i / 10 for i in range(11)]  # 0.0 to 1.0 in steps of 0.1
    results = []

    for alpha in alpha_values:
        print(f"Testing hybrid estimator with alpha = {alpha:.1f}")
        estimated_exposure = hybrid_estimator(graph, n, alpha)
        results.append((alpha, estimated_exposure))
    
    # Print results
    for alpha, exposure in results:
        print(f"Alpha: {alpha:.1f}, Estimated Exposure: {exposure:.4f}")
    
    # Plot the results (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        alphas, exposures = zip(*results)
        plt.plot(alphas, exposures, marker='o')
        plt.xlabel('Alpha')
        plt.ylabel('Estimated Exposure')
        plt.title('Effect of Alpha on Hybrid Estimator')
        plt.grid(True)
        plt.show()
    except ImportError:
        print("Matplotlib not installed. Install it to visualize the results.")