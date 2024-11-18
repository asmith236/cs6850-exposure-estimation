import os
import networkx as nx
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt

# dir containing the graph files
directory = "./facebook"

# init an empty graph
G = nx.Graph()

# iterate through the graph files and parse the data
for filename in os.listdir(directory):
    if filename.endswith(".edges"):
        ego_node = int(filename.split(".")[0])
        edges_path = os.path.join(directory, filename)

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
        feat_path = os.path.join(directory, filename)
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

# verify the graph structure
print("graph has been created with {} nodes and {} edges".format(G.number_of_nodes(), G.number_of_edges()))

# visualize full graph
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

# visualize full graph interactively w plotly
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

# visualize_graph(G)
visualize_interactive_plotly(G)
