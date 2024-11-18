import os
import networkx as nx
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

#TODO resolve missing nodes in init_graph

def init_graph(dir):
    """ visualizes the given graph using plotly for interactivity (can zoom in and out to view different clusters)
    
    returns:
    graph (nx.Graph): the network graph detailed in by the dir file information
    """
    # init an empty graph
    G = nx.Graph()

    # iterate through the graph files and parse the data
    for filename in os.listdir(dir):
        if filename.endswith(".edges"):
            ego_node = int(filename.split(".")[0])
            edges_path = os.path.join(dir, filename)

            # add ego node to the graph
            G.add_node(ego_node)

            # add edges to the graph
            with open(edges_path, 'r') as f:
                for line in f:
                    node1, node2 = map(int, line.strip().split())
                    G.add_edge(node1, node2)

            # connect ego node to its friends
            for friend in G.neighbors(ego_node):
                G.add_edge(ego_node, friend)

        elif filename.endswith(".feat"):
            feat_path = os.path.join(dir, filename)
            df = pd.read_csv(feat_path, delimiter=" ", header=None)

            # add features as node attributes
            for row in df.itertuples(index=False):
                node = row[0]
                features = row[1:]

                # add the node if it does not already exist
                if node not in G:
                    G.add_node(node)

                # add features to the node
                G.nodes[node]['features'] = features

    # verify graph structure
    print("graph has been created with {} nodes and {} edges".format(G.number_of_nodes(), G.number_of_edges()))
    return G

def visualize_graph(graph):
    """ visualizes given graph as an image using networkx and matplotlib
    
    params:
    graph (nx.Graph): the graph to be visualized
    """
    # draw full graph
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(graph, seed=42)  # layout for nodes
    nx.draw(graph, pos, with_labels=False, node_color='skyblue', edge_color='gray', node_size=50, font_size=8, font_weight='bold')
    plt.title("Facebook Ego Network")
    plt.show()

def visualize_interactive_plotly(graph):
    """ visualizes the given graph using plotly for interactivity (can zoom in and out to view different clusters)
    
    params:
    graph (nx.Graph): the graph to be visualized
    """
    # Create position layout
    pos = nx.spring_layout(graph, seed=42)

    # Extract edges and nodes
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(graph.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'Node {adjacencies[0]} has {len(adjacencies[1])} connections')

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Interactive Visualization of the Facebook Ego Network',
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    fig.show()

def assign_node_quality_dist(graph, distribution='normal', **kwargs):
    """ assigns each node in the graph a quality value drawn from a specified distribution
    
    params:
    graph (nx.Graph): the graph whose nodes will be assigned quality values
    distribution (str): the type of distribution to use (normal or uniform)
    kwargs: additional params for the distribution
    
    """
    if distribution == 'normal':
        mean = kwargs.get('mean', 0)
        std = kwargs.get('std', 1)
        qualities = np.random.normal(mean, std, graph.number_of_nodes())
    elif distribution == 'uniform':
        low = kwargs.get('low', 0)
        high = kwargs.get('high', 1)
        qualities = np.random.uniform(low, high, graph.number_of_nodes())
    else:
        raise ValueError("unsupported distribution type; use 'normal' or 'uniform'")

    for i, node in enumerate(graph.nodes()):
        graph.nodes[node]['quality'] = qualities[i]

def assign_node_quality_prop(graph):
    """ assigns each node in the graph a quality value derived from graph properties
    
    params:
    graph (nx.Graph): the graph whose nodes will be assigned quality values
    """
    quality_values = []
    for node in graph.nodes():
        # calc quality as a fn of node features and clustering coefficient
        clustering_coefficient = nx.clustering(graph, node)
        degree_centrality = nx.degree_centrality(graph)[node]
        features = graph.nodes[node].get('features', [])
        feature_score = sum(features) / len(features) if len(features) > 0 else 1  # average feature value or default to 1

        # derive quality based on clustering, degree centrality, and feature score
        quality = round((0.5 * clustering_coefficient) + (0.3 * degree_centrality) + (0.2 * feature_score), 2)
        graph.nodes[node]['quality'] = quality
        quality_values.append(quality)

    # count the number of nodes with each quality value
    quality_counts = Counter(quality_values)
    # for quality, count in sorted(quality_counts.items()):
    #     print(f'quality: {quality:.2f}, count: {count}')

    # plot the distribution of quality values
    plt.figure(figsize=(10, 6))
    plt.bar(quality_counts.keys(), quality_counts.values(), color='skyblue')
    plt.xlabel('Quality Value')
    plt.ylabel('Number of Nodes')
    plt.title('Distribution of Node Quality Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('quality_distribution.png')

    # verify all nodes in graph have an assigned quality
    if len(quality_values) == graph.number_of_nodes():
        print("quality assigned to all nodes")
    else:
        print("error; the num of nodes assigned a quality value does not match the total number of nodes in the graph")

