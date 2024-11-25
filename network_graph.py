import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

# TODO determine assortativity coefficient of graph
class GraphModel:
    def __init__(self, num_nodes=10000, alpha=2.5, p_share=0.5, seed=None):
        """
        Initializes the GraphModel based on the configuration model for a power-law degree distribution.

        Parameters:
        - num_nodes: Number of nodes in the graph.
        - alpha: Power-law exponent for degree distribution.
        - p_share: Probability of sharing information (for the Bernoulli distribution).
        - seed: Random seed for reproducibility.
        """
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.p_share = p_share
        self.seed = seed
        self.G = self.generate_graph()
        self.assign_sharing_function()
        self.true_average_exposure = self.calculate_true_average_exposure()
        # TODO uncomment to mimic FB social network
        # self.assortativity = self.set_assortativity() # Set assortativity to zero to mimic Facebook social network
        # self.correlation = self.set_correlation() # Set correlation to 0.2 to mimic Facebook social network

    def generate_graph(self):
        """
        Generate a graph using the configuration model with a power-law degree distribution.
        """
        np.random.seed(self.seed)
        degree_sequence = np.random.zipf(self.alpha, size=self.num_nodes)
        degree_sequence = degree_sequence[degree_sequence > 1]  # Remove nodes with degree <= 1 for connectivity

        # Ensure the sum of the degree sequence is even
        if sum(degree_sequence) % 2 != 0:
            degree_sequence[-1] += 1

        G = nx.configuration_model(degree_sequence, seed=self.seed)
        G = nx.Graph(G)  # Remove parallel edges and self-loops
        G.remove_edges_from(nx.selfloop_edges(G))
        print(f"Generated graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return G

    def assign_sharing_function(self):
        """
        Assign sharing function values (s(v)) as iid Bernoulli random variables to each node.
        """
        for node in self.G.nodes:
            self.G.nodes[node]["share"] = np.random.binomial(1, self.p_share)

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
    
    def visualize_graph(self, num_nodes=500, layout="spring", node_color=None):
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
    def set_assortativity(self, target_r=0, max_iterations=10000, reset=False):
        """
        Adjust the graph's assortativity coefficient (r) to the desired target value.

        Parameters:
        - target_r: Desired assortativity coefficient (-1 <= target_r <= 1).
        - max_iterations: Maximum number of edge rewiring attempts to achieve the target.

        Returns:
        - r: The assortativity coefficient after rewiring.
        """
        current_r = nx.degree_assortativity_coefficient(self.G)
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
    def set_correlation(self, target_correlation=0.2, max_iterations=100000):
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
    graph_model = GraphModel(num_nodes=1000, alpha=2.5, p_share=0.5, seed=42)

    # Visualize the graph
    graph_model.visualize_graph(num_nodes=200, layout="kamada_kawai")
