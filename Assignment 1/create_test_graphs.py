import networkx as nx
import numpy as np

G = nx.random_tree(50)

K = 5
G.graph['K'] = K

for node in G.nodes:
	G.nodes[node]['unary_potential'] = np.random.rand(K,)
	G.nodes[node]['assignment'] = 1

for edge in G.edges:
	G.edges[edge]['binary_potential'] = np.random.rand(K, K)

nx.write_gpickle(G, "./graph50.pickle")
