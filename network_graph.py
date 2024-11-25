# import os
# import networkx as nx
# import pandas as pd
# import plotly.graph_objs as go
# import matplotlib.pyplot as plt
# import numpy as np
# from collections import Counter

# #TODO resolve missing nodes in init_graph

# class GraphModel:
#     def __init__(self, graph_dir, rho_type='uniform', bins=10):
#         """
#         inits the GraphModel with a graph and params
        
#         params:
#         - graph_dir (str): directory containing graph data files
#         - rho_type (str): type of quality distribution ('uniform' or 'empirical')
#         - bins (int): number of bins for empirical rho(theta) if using 'empirical'
#         """
#         self.graph = self.init_graph_aaron()
#         # self.graph = self.init_graph(graph_dir)

#         self.beta = self.compute_beta()
#         self.assign_node_quality_uniform()
#         self.mu = self.compute_mu()
        
#         if rho_type == 'uniform':
#             self.rho = self.rho_uniform
#         elif rho_type == 'empirical':
#             self.rho = self.rho_empirical(bins=bins)
#         else:
#             raise ValueError("invalid rho_type, choose 'uniform' or 'empirical'")
        
#         print(f"graph model initialized with beta={self.beta}, mu={self.mu:.4f}")

#     def init_graph_aaron(self, size=2500):
#         """ initializes a graph with 10 nodes and 15 edges
        
#         returns:
#         graph (nx.Graph): the network graph detailed in by the dir file information
#         """
#         # init an empty graph
#         G = nx.Graph()

#         # add nodes
#         for i in range(size):
#             G.add_node(i)

#         # add edges
#         for i in range(size):
#             degree = np.random.randint(1, size)
#             for _ in range(degree):
#                 neighbor = np.random.randint(0, size)
#                 G.add_edge(i, neighbor)

#         print("graph has been created with {} nodes and {} edges".format(G.number_of_nodes(), G.number_of_edges()))
#         return G

    
#     def init_graph(self, dir):
#         """
#         inits the graph from the given dir
#         """
#         G = nx.Graph()
#         for filename in os.listdir(dir):
#             if filename.endswith(".edges"):
#                 ego_node = int(filename.split(".")[0])
#                 edges_path = os.path.join(dir, filename)
#                 G.add_node(ego_node)
#                 with open(edges_path, 'r') as f:
#                     for line in f:
#                         node1, node2 = map(int, line.strip().split())
#                         G.add_edge(node1, node2)
#                 for friend in G.neighbors(ego_node):
#                     G.add_edge(ego_node, friend)
#             elif filename.endswith(".feat"):
#                 feat_path = os.path.join(dir, filename)
#                 df = pd.read_csv(feat_path, delimiter=" ", header=None)
#                 for row in df.itertuples(index=False):
#                     node = row[0]
#                     features = row[1:]
#                     if node not in G:
#                         G.add_node(node)
#                     G.nodes[node]['features'] = features
#         print("Graph has been created with {} nodes and {} edges".format(G.number_of_nodes(), G.number_of_edges()))
#         return G

#     def compute_beta(self):
#         """
#         compute beta as the minimum degree in the graph
#         """
#         min_degree = min(dict(self.graph.degree()).values())
#         return max(1, min_degree)

#     def assign_node_quality_uniform(self):
#         """
#         assign random quality values between 0.0 and 1.0 to each node
#         """
#         for node in self.graph.nodes:
#             self.graph.nodes[node]['quality'] = np.random.uniform(0.0, 1.0)

#     def compute_mu(self):
#         """
#         compute the average quality (mu) of the nodes
#         """
#         qualities = [self.graph.nodes[node]['quality'] for node in self.graph.nodes]
#         return np.mean(qualities)

#     def rho_uniform(self, theta):
#         """
#         define rho(theta) as a uniform distribution over [0, 1]
#         """
#         return 1.0 if 0.0 <= theta <= 1.0 else 0.0

#     def rho_empirical(self, bins=10):
#         """
#         compute an empirical distribution of qualities
#         """
#         qualities = [self.graph.nodes[node]['quality'] for node in self.graph.nodes]
#         hist, bin_edges = np.histogram(qualities, bins=bins, density=True)
        
#         def rho(theta):
#             for i in range(len(bin_edges) - 1):
#                 if bin_edges[i] <= theta < bin_edges[i + 1]:
#                     return hist[i]
#             return 0.0
        
#         return rho

# def visualize_graph(graph):
#     """ visualizes given graph as an image using networkx and matplotlib
    
#     params:
#     graph (nx.Graph): the graph to be visualized
#     """
#     # draw full graph
#     plt.figure(figsize=(12, 12))
#     pos = nx.spring_layout(graph, seed=42)  # layout for nodes
#     nx.draw(graph, pos, with_labels=False, node_color='skyblue', edge_color='gray', node_size=50, font_size=8, font_weight='bold')
#     plt.title("Facebook Ego Network")
#     plt.show()

# def visualize_interactive_plotly(graph):
#     """ visualizes the given graph using plotly for interactivity (can zoom in and out to view different clusters)
    
#     params:
#     graph (nx.Graph): the graph to be visualized
#     """
#     # create position layout
#     pos = nx.spring_layout(graph, seed=42)

#     # extract edges and nodes
#     edge_x = []
#     edge_y = []
#     for edge in graph.edges():
#         x0, y0 = pos[edge[0]]
#         x1, y1 = pos[edge[1]]
#         edge_x.extend([x0, x1, None])
#         edge_y.extend([y0, y1, None])

#     edge_trace = go.Scatter(
#         x=edge_x,
#         y=edge_y,
#         line=dict(width=0.5, color='#888'),
#         hoverinfo='none',
#         mode='lines')

#     node_x = []
#     node_y = []
#     for node in graph.nodes():
#         x, y = pos[node]
#         node_x.append(x)
#         node_y.append(y)

#     node_trace = go.Scatter(
#         x=node_x,
#         y=node_y,
#         mode='markers',
#         hoverinfo='text',
#         marker=dict(
#             showscale=True,
#             colorscale='YlGnBu',
#             reversescale=True,
#             color=[],
#             size=10,
#             colorbar=dict(
#                 thickness=15,
#                 title='Node Connections',
#                 xanchor='left',
#                 titleside='right'
#             ),
#             line_width=2))

#     node_adjacencies = []
#     node_text = []
#     for node, adjacencies in enumerate(graph.adjacency()):
#         node_adjacencies.append(len(adjacencies[1]))
#         node_text.append(f'Node {adjacencies[0]} has {len(adjacencies[1])} connections')

#     node_trace.marker.color = node_adjacencies
#     node_trace.text = node_text

#     fig = go.Figure(data=[edge_trace, node_trace],
#                     layout=go.Layout(
#                         title='Interactive Visualization of the Facebook Ego Network',
#                         titlefont=dict(size=16),
#                         showlegend=False,
#                         hovermode='closest',
#                         margin=dict(b=0, l=0, r=0, t=0),
#                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

#     fig.show()


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
