import networkx as nx
import matplotlib.pylab as plt
import random
import numpy as np

import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def chinese_whispers(ordered_edges, labels, num_iterations, rnd):
    G = nx.Graph()
    #G.add_nodes_from(nodes)
    G.add_edges_from(ordered_edges)
    #clear the list
    labels = range(len(ordered_edges))

    if len(ordered_edges) == 0:
        return 0

    neighbors = [G[i] for i in G.nodes()]

    for iter in range(len(ordered_edges) * num_iterations):
        idx1 = np.random.randint(0, high=len(ordered_edges))

        labels_to_counts = {}

        for idx2 in neighbors[idx1]:
            labels_to_counts[labels[idx2]] += G[idx1][idx2]['distance']

        #find the most common label
        best_label = labels[idx1]
        bset_score = -1
        for k,v in  labels_to_counts.iteritems():
            best_score = v
            best_label = k 

        labels[idx1] = best_label







    
    

nodes = [
        (0, {'attr1':1, 'class':0}),
        (1, {'attr1':1, 'class':0}),
        (2, {'attr1':1, 'class':0})
        ]

edges = [
        (1,2,{'weight':0.732})
        ]

#initialize the graph

G = nx.Graph()

G.add_nodes_from(nodes)

for n, v in enumerate(nodes):
    print n, v
    G.node[n]['class'] = n

G.add_edges_from(edges) 

nx.draw(G)
plt.show()

print 'edges type:', type(G.edges())


#run Chinese Whisper
iterations = 10

for z in range(0, iterations):
    gn = G.nodes()
    random.shuffle(gn)
    for node in gn:
        neighs = G[node]
        print 'node:', node, ',neighs:', neighs
        classes = {}
        #do an inventory of the given nodes and edge weights
        for ne in neighs:
            if isinstance(ne, int):
                if G.node[ne]['class'] in classes:
                    classes[G.node[ne]['class']] += G[node][ne]['weight']
                else:
                    classes[G.node[ne]['class']] = G[node][ne]['weight']
        #find the class with the highest edge weight sum
        max_n = 0
        maxclass = 0
        for c in classes:
            if classes[c] > max_n:
                max_n = classes[c]
                maxclass = c
        #set the class of target node to the winning local class
        G.node[node]['class'] = maxclass

                    
