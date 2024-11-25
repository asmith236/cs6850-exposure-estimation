import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

class GraphModel:
    def __init__(self, init_num_nodes=10000, alpha=2.5, p_share=0.5, seed=None):
        """
        Initializes the GraphModel based on the configuration model for a power-law degree distribution.

        Parameters:
        - num_nodes: Number of nodes in the graph before connectivity assurance.
        - alpha: Power-law exponent for degree distribution.
        - p_share: Probability of sharing information (for the Bernoulli distribution).
        - seed: Random seed for reproducibility.
        """
        self.alpha = alpha
        self.p_share = p_share
        self.seed = seed
        self.G = self.generate_graph(init_num_nodes)
        self.num_nodes = self.G.number_of_nodes()
        self.beta = self.compute_beta()  # Compute beta after generating the graph
        self.assign_sharing_function()
        self.assign_node_quality()
        self.mu = self.compute_mu()
        self.rho = self.define_rho()  # Define rho(θ)
        self.true_average_exposure = self.calculate_true_average_exposure()
        # Note: Can toggle assortativity and correlation to mimic different networks
        self.assortativity = self.set_assortativity(reset=False) # TODO Set assortativity to zero to mimic Facebook social network
        self.correlation = self.set_correlation(reset=False) # TODO Set correlation to 0.2 to mimic Facebook social network
        print(f"Created graph with {self.num_nodes} nodes, {self.G.number_of_edges()} edges, and the following params:")
        print(f"alpha = {self.alpha}")
        print(f"p_share = {self.p_share}")
        print(f"seed = {self.seed}")
        print(f"assortativity = {self.assortativity}")
        print(f"correlation = {self.correlation}")
        print(f"mu = {self.mu}")
        print(f"beta = {self.beta}")
        print(f"rho = {self.beta}")
        print(f"true avg exposure = {self.true_average_exposure}")

    # Note: Power-law degree distributions have a heavy tailed distributions. This means that most nodes have relatively 
    # few connections (small k), but a few nodes have extremely high degrees (hubs), which mimics most social networks
    def generate_graph(self, init_num_nodes):
        """
        Generate a graph using the configuration model with a power-law degree distribution.
        """
        np.random.seed(self.seed)
        degree_sequence = np.random.zipf(self.alpha, size=init_num_nodes)
        degree_sequence = degree_sequence[degree_sequence > 1]  # Remove nodes with degree <= 1 for connectivity
        # Ensure the sum of the degree sequence is even
        if sum(degree_sequence) % 2 != 0:
            degree_sequence[-1] += 1

        G = nx.configuration_model(degree_sequence, seed=self.seed)
        G = nx.Graph(G)  # Remove parallel edges and self-loops
        G.remove_edges_from(nx.selfloop_edges(G))
        return G
    
    # TODO verify that beta aligns with theoretical assumptions
    def compute_beta(self):
        """
        Compute beta as the minimum degree in the graph.
        """
        min_degree = min(dict(self.G.degree()).values())
        return max(1, min_degree)  # Ensure beta is at least 1

    def assign_sharing_function(self):
        """
        Assign sharing function values (s(v)) as iid Bernoulli random variables to each node.
        """
        for node in self.G.nodes:
            self.G.nodes[node]["share"] = np.random.binomial(1, self.p_share)
    
    def assign_node_quality(self):
        """
        Assign quality values (theta) as uniform random values between 0 and 1.0 to each node.
        """
        for node in self.G.nodes:
            self.G.nodes[node]["quality"] = np.random.uniform(0.0, 1.0)
    
    def compute_mu(self):
        """
        Compute the mean quality value µ as the average of node qualities.
        """
        qualities = [self.G.nodes[node]["quality"] for node in self.G.nodes]
        return np.mean(qualities)
    
    def define_rho(self):
        """
        Define the probability density function ρ(θ) for node qualities.
        Since θ follows a uniform distribution over [0, 1], ρ(θ) = 1 for θ ∈ [0, 1].
        """
        def rho(theta):
            if 0 <= theta <= 1:
                return 1  # Uniform distribution over [0, 1]
            else:
                return 0  # Out of range
        return rho

    def calculate_true_average_exposure(self):
        """
        Calculate the true average exposure (ground truth) as the mean of sharing function values.
        """
        share_values = [self.G.nodes[node]["share"] for node in self.G.nodes]
        return np.mean(share_values)

    def average_degree(self):
        """
        Calculate the average degree of the graph.
        """
        total_degree = sum(dict(self.G.degree).values())
        return total_degree / self.G.number_of_nodes()
    
    def visualize_graph(self, num_nodes=10000, layout="spring", node_color=None):
        """
        Visualize the graph using matplotlib and networkx.

        Parameters:
        - num_nodes: Maximum number of nodes to visualize (useful for large graphs).
        - layout: Layout algorithm for positioning nodes ('spring', 'circular', etc.).
        - node_color: Node coloring based on an attribute (default is None, which assigns random colors).
        """
        # Subset the graph if it's too large for visualization
        if len(self.G.nodes) > num_nodes:
            sampled_nodes = random.sample(self.G.nodes, num_nodes)
            subgraph = self.G.subgraph(sampled_nodes)
        else:
            subgraph = self.G

        # Choose the layout for visualization
        if layout == "spring":
            pos = nx.spring_layout(subgraph, seed=self.seed)  # Spring layout (force-directed)
        elif layout == "circular":
            pos = nx.circular_layout(subgraph)  # Circular layout
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(subgraph)  # Kamada-Kawai layout
        else:
            raise ValueError(f"Unknown layout: {layout}")

        # Node color based on the sharing function values
        if node_color is None:
            node_color = [subgraph.nodes[node]["share"] for node in subgraph.nodes]

        # Plot the graph
        plt.figure(figsize=(10, 8))
        nx.draw(
            subgraph,
            pos,
            with_labels=False,
            node_size=50,
            node_color=node_color,
            cmap=plt.cm.viridis,
            edge_color="gray",
            alpha=0.7
        )
        plt.title(f"Graph Visualization ({len(subgraph.nodes)} nodes, {len(subgraph.edges)} edges)")
        plt.show()

    # Note: A "good" assortativity coefficient for a Facebook social network would be considered 
    # close to zero, indicating a mostly neutral assortativity, meaning that users tend to 
    # connect with others who have a similar number of friends, but not significantly more or less 
    # so than would be expected by chance.
    def set_assortativity(self, reset, target_r=0.10, max_iterations=10000):
        """
        Adjust the graph's assortativity coefficient (r) to the desired target value.

        Parameters:
        - target_r: Desired assortativity coefficient (-1 <= target_r <= 1).
        - max_iterations: Maximum number of edge rewiring attempts to achieve the target.

        Returns:
        - r: The assortativity coefficient after rewiring.
        """
        current_r = nx.degree_assortativity_coefficient(self.G)

        if reset == False:
            return current_r
        
        print(f"Initial assortativity coefficient: {current_r:.4f}")
        # Rewire edges to achieve the target assortativity
        iterations = 0
        while abs(current_r - target_r) > 0.01 and iterations < max_iterations:
            # Randomly select two edges
            edges = list(self.G.edges)
            u, v = random.choice(edges)
            x, y = random.choice(edges)

            # Ensure no self-loops or duplicate edges
            if len({u, v, x, y}) < 4:
                continue

            # Swap edges to increase/decrease assortativity
            if random.random() < 0.5:
                # Connect u-x and v-y
                self.G.remove_edge(u, v)
                self.G.remove_edge(x, y)
                self.G.add_edge(u, x)
                self.G.add_edge(v, y)
            else:
                # Connect u-y and v-x
                self.G.remove_edge(u, v)
                self.G.remove_edge(x, y)
                self.G.add_edge(u, y)
                self.G.add_edge(v, x)

            # Recalculate assortativity
            current_r = nx.degree_assortativity_coefficient(self.G)
            iterations += 1

        print(f"Final assortativity coefficient: {current_r:.4f} after {iterations} iterations.")
        return current_r
    
    # Note: A "good" correlation coefficient for a Facebook social network, depending on the analysis 
    # being done, would generally fall in the range of 0.2 to 0.5 which indicates a moderate positive 
    # correlation, as social networks tend to exhibit complex relationships with not all users strongly 
    # connected to each other.
    def set_correlation(self, reset, target_correlation=0.1, max_iterations=10000):
        """
        Adjust the degree-sharing correlation coefficient of the graph to the desired target value.

        Parameters:
        - target_correlation: Desired degree-sharing correlation coefficient (-1 <= target_correlation <= 1).
        - max_iterations: Maximum number of edge rewiring attempts to achieve the target.

        Returns:
        - corr: The degree-sharing correlation coefficient after rewiring.
        """
        def calculate_correlation(G):
            """Calculate the degree-sharing correlation coefficient."""
            degrees = dict(G.degree())
            sharing_values = nx.get_node_attributes(G, "share")

            degree_sharing_pairs = [(degrees[node], sharing_values[node]) for node in G.nodes]
            degrees, shares = zip(*degree_sharing_pairs)

            return np.corrcoef(degrees, shares)[0, 1]  # Pearson correlation coefficient

        # Calculate the current degree-sharing correlation
        current_correlation = calculate_correlation(self.G)

        if reset == False:
            return current_correlation
        
        print(f"Initial degree-sharing correlation: {current_correlation:.4f}")
        # Rewire edges to adjust the correlation
        iterations = 0
        while abs(current_correlation - target_correlation) > 0.01 and iterations < max_iterations:
            # Randomly select two edges
            edges = list(self.G.edges)
            u, v = random.choice(edges)
            x, y = random.choice(edges)

            # Ensure no self-loops or duplicate edges
            if len({u, v, x, y}) < 4:
                continue

            # Rewire edges
            if random.random() < 0.5:
                # Connect u-x and v-y
                self.G.remove_edge(u, v)
                self.G.remove_edge(x, y)
                self.G.add_edge(u, x)
                self.G.add_edge(v, y)
            else:
                # Connect u-y and v-x
                self.G.remove_edge(u, v)
                self.G.remove_edge(x, y)
                self.G.add_edge(u, y)
                self.G.add_edge(v, x)

            # Recalculate degree-sharing correlation
            current_correlation = calculate_correlation(self.G)
            iterations += 1

        print(f"Final degree-sharing correlation: {current_correlation:.4f} after {iterations} iterations.")
        return current_correlation

if __name__ == "__main__":
    # Initialize the graph model
    graph_model = GraphModel(init_num_nodes=10000, alpha=2.5, p_share=0.5, seed=42)

    # Visualize the graph
    graph_model.visualize_graph(num_nodes=1000, layout="kamada_kawai")
