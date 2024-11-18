import network as n
from collections import defaultdict

# dqjd-inspired estimator 
def dqjd_estimator(graph):
    """ implements the dqjd-inspired estimator
    
    uses the joint distribution of degree and quality to estimate the probability that a node is exposed
    
    params:
    graph (nx.Graph): the graph for which the estimation is performed.
    
    returns:
    float: estimated fraction of exposed nodes in the network.
    """
    # create dicts to hold degree and quality distribs
    degree_counts = defaultdict(int)
    quality_counts = defaultdict(int)
    joint_counts = defaultdict(lambda: defaultdict(int))

    # gather degree and quality data
    for node in graph.nodes():
        degree = graph.degree[node]
        quality = graph.nodes[node]['quality']

        degree_counts[degree] += 1
        quality_counts[quality] += 1
        joint_counts[degree][quality] += 1

    # calc total num nodes
    total_nodes = graph.number_of_nodes()

    # calc P(k, θ) using the dqjd
    p_k_theta = {}
    for degree in joint_counts:
        p_k_theta[degree] = {}
        for quality in joint_counts[degree]:
            p_k_theta[degree][quality] = joint_counts[degree][quality] / total_nodes

    # define f(k, θ) - the exposure prob fn based on degree and quality
    def f_k_theta(degree, quality):
        # ex - higher degree and quality nodes have a higher exposure probability
        return min(1, 0.5 + 0.05 * degree + 0.1 * quality)

    # estimate expected fraction of exposed nodes
    estimated_exposure = 0
    for degree in p_k_theta:
        for quality in p_k_theta[degree]:
            estimated_exposure += p_k_theta[degree][quality] * f_k_theta(degree, quality)

    print(f"estimated fraction of exposed nodes = {estimated_exposure:.4f}")
    return estimated_exposure
