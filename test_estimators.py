import network as n

# dir containing the graph files
dir = "./facebook"

G = n.init_graph(dir)

# assign quality values to nodes 
# n.assign_node_quality_dist(G, distribution='normal', mean=0, std=1)
n.assign_node_quality_prop(G)

# n.visualize_interactive_plotly(G)

