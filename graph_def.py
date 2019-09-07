import itertools
import copy
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

nodelist = pd.read_csv(
    'https://gist.githubusercontent.com/brooksandrew/f989e10af17fb4c85b11409fea47895b/raw/a3a8da0fa5b094f1ca9d82e1642b384889ae16e8/nodelist_sleeping_giant.csv')
edgelist = pd.read_csv(
    'https://gist.githubusercontent.com/brooksandrew/e570c38bcc72a8d102422f2af836513b/raw/89c76b2563dbc0e88384719a35cba0dfc04cd522/edgelist_sleeping_giant.csv')
edgelist.head(10)
nodelist.head(5)

# Create empty graph
g = nx.Graph()

# Add edges and edge attributes
for i, elrow in edgelist.iterrows():
    g.add_edge(elrow[0], elrow[1], **elrow[2:].to_dict())

# To illustrate what happened in the above iteration let me just print the values from the last row in the edge list
# that got added to graph g:
print(elrow[0])  # node1
print(elrow[1])  # node2
print(elrow[2:].to_dict())  # edge attribute dict

# Adding node attributes
for i, nlrow in nodelist.iterrows():
    nx.set_node_attributes(g, {nlrow['id']: nlrow[1:].to_dict()})

# Node list example
print(nlrow)

# Preview first 5 edges
list(g.edges(data=True))[0:5]
# Preview first 5 nodes
list(g.nodes(data=True))[0:10]

# Print out some summary statistics before visualizing the graph.
print('# of edges: {}'.format(g.number_of_edges()))
print('# of nodes: {}'.format(g.number_of_nodes()))

# Define node positions data structure (dict) for plotting
node_positions = {node[0]: (node[1]["X"], -node[1]["Y"]) for node in g.nodes(data=True)}

# Preview of node_positions with a bit of hack (there is no head/slice method for dictionaries).
dict(list(node_positions.items())[0:5])

# Define data structure (list) of edge colors for plotting
edge_colors = [e[2]['color'] for e in list(g.edges(data=True))]

# Preview first 10
edge_colors[0:10]

plt.figure(figsize=(8, 6))
nx.draw(g, pos=node_positions, edge_color=edge_colors, node_size=10, node_color='black')
plt.title('Graph Representation of a simple program', size=15)
plt.show()

# Calculate list of nodes with odd degree (death ends)
nodes_odd_degree = [v for v, d in g.degree() if d % 2 == 1]

# Preview
nodes_odd_degree[0:5]

print('Number of nodes of odd degree: {}'.format(len(nodes_odd_degree)))
print('Number of total nodes: {}'.format(len(g.nodes())))

# Compute all pairs of odd nodes. in a list of tuples
odd_node_pairs = list(itertools.combinations(nodes_odd_degree, 2))

# Preview pairs of odd degree nodes
odd_node_pairs[0:10]

# Counts
print('Number of pairs: {}'.format(len(odd_node_pairs)))


def get_shortest_paths_distances(graph, pairs, edge_weight_name):
    """Compute shortest distance between each pair of nodes in a graph.  Return a dictionary keyed on node pairs (
    tuples). """
    distances = {}
    for pair in pairs:
        distances[pair] = nx.dijkstra_path_length(graph, pair[0], pair[1], weight=edge_weight_name)
    return distances


# Compute shortest paths.  Return a dictionary with node pairs keys and a single value equal to shortest path distance.
odd_node_pairs_shortest_paths = get_shortest_paths_distances(g, odd_node_pairs, 'distance')

# Preview with a bit of hack (there is no head/slice method for dictionaries).
dict(list(odd_node_pairs_shortest_paths.items())[0:10])
