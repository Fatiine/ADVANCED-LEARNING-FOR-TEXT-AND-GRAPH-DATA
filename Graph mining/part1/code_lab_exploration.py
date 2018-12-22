"""
Graph Mining - ALTEGRAD - Dec 2018
"""

# Import modules
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


############## Question 1
# Load the graph into an undirected NetworkX graph

G=nx.read_edgelist("../datasets/CA-HepTh.txt", comments='#', delimiter='\t', nodetype=int, create_using=nx.Graph())
   


############## Question 2
# Network Characteristics


print ('The number of nodes:', G.number_of_nodes() )
print ('The number of edges:', G.number_of_edges() )
print ('The number of connected components:', nx.number_connected_components(G))

# The largest connected component (GCC)
GCC=list(nx.connected_component_subgraphs(G))[0]

# Number of its nodes and edges 
print ('The number of nodes in GCC', GCC.number_of_nodes())
print ('The number of edges in GCC', GCC.number_of_edges())


# fraction of the whole graph they correspond
print ('Fraction of nodes in GCC', GCC.number_of_nodes() / G.number_of_nodes())
print ('Fraction of edges in GCC', GCC.number_of_edges() / G.number_of_edges())


############## Question 3
# Analysis of degree distribution

degree_sequence = [d for n, d in G. degree ()]

print ('Min degree', np.min(degree_sequence))
print ('Max degree', np.max(degree_sequence))
print ('Mean degree', np.mean(degree_sequence))


y=nx.degree_histogram(G)

plt.figure(1)
plt.plot(y,'b',marker='o')
plt.ylabel("Frequency")
plt.xlabel("Degree")
plt.show()

plt.figure(2)
plt.loglog(y,'b',marker='o')
plt.ylabel("Frequency")
plt.xlabel("Degree")
plt.show()
