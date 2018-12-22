"""
Graph Mining - ALTEGRAD - Dec 2018
"""

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from grakel.kernels import ShortestPath, PyramidMatch, RandomWalk, VertexHistogram, WeisfeilerLehman
from grakel import graph_from_networkx
from grakel.datasets import fetch_dataset
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


############## Question 1
# Generate simple dataset

Gs = list()
y = list()

for n in range(3, 103):
    Gs.append(nx.path_graph(n))
    y.append(0)
    Gs.append(nx.cycle_graph(n))
    y.append(1)


############## Question 2
# Classify the synthetic graphs using graph kernels

# Split dataset into a training and a test set
# hint: use the train_test_split function of scikit-learn

G_train, G_test, y_train, y_test = train_test_split(Gs, y, test_size=0.1)

# Transform NetworkX graphs to objects that can be processed by GraKeL
G_train = list(graph_from_networkx(G_train))
G_test = list(graph_from_networkx(G_test))


# Use the shortest path kernel to generate the two kernel matrices ("K_train" and "K_test")
# hint: the graphs do not contain node labels. Set the with_labels argument of the the shortest path kernel to False

gk = ShortestPath(with_labels=False)

K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)


clf = SVC(kernel='precomputed', C=1) # Initialize SVM
clf.fit(K_train, y_train) # Train SVM
y_pred = clf.predict(K_test) # Predict

# Compute the classification accuracy
# hint: use the accuracy_score function of scikit-learn

print("Classification accuracy using ShortestPath", accuracy_score(y_test, y_pred))



# Use the random walk kernel and the pyramid match graph kernel to perform classification

gk = RandomWalk()

K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)

clf = SVC(kernel='precomputed', C=1) # Initialize SVM
clf.fit(K_train, y_train) # Train SVM
y_pred = clf.predict(K_test) # Predict

print("Classification accuracy using RandomWalk", accuracy_score(y_test, y_pred))

gk = PyramidMatch(with_labels=False)

K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)

clf = SVC(kernel='precomputed', C=1) # Initialize SVM
clf.fit(K_train, y_train) # Train SVM
y_pred = clf.predict(K_test) # Predict

print("Classification accuracy using PyramidMatch", accuracy_score(y_test, y_pred))



############## Question 3
# Classify the graphs of a real-world dataset using graph kernels

# Load the MUTAG dataset
# hint: use the fetch_dataset function of GraKeL

mutag = fetch_dataset("MUTAG", verbose = False)
G, y = mutag.data, mutag.target

# Split dataset into a training and a test set
# hint: use the train_test_split function of scikit-learn

G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.1)

# Perform graph classification using different kernels and evaluate performance

gk = VertexHistogram()

K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)

clf = SVC(kernel='precomputed', C=1) # Initialize SVM
clf.fit(K_train, y_train) # Train SVM
y_pred = clf.predict(K_test) # Predict

print("Classification accuracy using VertexHistogram", accuracy_score(y_test, y_pred))


gk = ShortestPath(with_labels=True)

K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)

clf = SVC(kernel='precomputed', C=1) # Initialize SVM
clf.fit(K_train, y_train) # Train SVM
y_pred = clf.predict(K_test) # Predict

print("Classification accuracy using ShortestPath", accuracy_score(y_test, y_pred))

gk = PyramidMatch(with_labels=True)

K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)

clf = SVC(kernel='precomputed', C=1) # Initialize SVM
clf.fit(K_train, y_train) # Train SVM
y_pred = clf.predict(K_test) # Predict

print("Classification accuracy using PyramidMatch", accuracy_score(y_test, y_pred))


gk = WeisfeilerLehman(base_kernel=VertexHistogram)

K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)

clf = SVC(kernel='precomputed', C=1) # Initialize SVM
clf.fit(K_train, y_train) # Train SVM
y_pred = clf.predict(K_test) # Predict

print("Classification accuracy using WeisfeilerLehman", accuracy_score(y_test, y_pred))
