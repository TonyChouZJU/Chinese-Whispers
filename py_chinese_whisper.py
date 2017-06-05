import networkx as nx
import matplotlib.pylab as plt
import random
import numpy as np
from collections import defaultdict

import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def convert_unordered_to_ordered(edges):
    assert len(edges) != 0
    #reserve
    out_edges = [()] * (2*len(edges))
    pend = 0 
    for edge in edges:
        out_edges[pend] = edge
        pend += 1
        if edge[0] != edge[1]:
            out_edges[pend] = (edge[1], edge[0], edge[2])
            pend += 1

    return out_edges[:pend]

#edges should be something like this
#[(1,2,5),(2,3,3})]
#or
#[(1,2,{'weight':5}),(2,3,{'weight':3})]
def chinese_whispers_ordered(ordered_edges, num_iterations):
    G = nx.Graph()
    #G.add_nodes_from(nodes)
    G.add_weighted_edges_from(ordered_edges)
    #clear the list

    neighbors = [G[i] for i in G.nodes()]
    logging.info('neighbors:{}'.format(neighbors))
    logging.info('neighbors len:{}'.format(len(neighbors)))

    if len(neighbors) == 0:
        return 0

    labels = range(len(neighbors))

    for iter in range(len(neighbors) * num_iterations):
        idx1 = np.random.randint(0, high=len(neighbors))

        #labels_to_counts = {}
        labels_to_counts = defaultdict(float)

        for idx2 in neighbors[idx1]:
            logging.info('idx2:{}'.format(idx2))
            logging.info('G[{}][{}][weight] is {}'.format(idx1, idx2, G[idx1][idx2]['weight']))
            labels_to_counts[labels[idx2]] += G[idx1][idx2]['weight']

        #find the most common label
        best_label = labels[idx1]
        best_score = -1.0
        for k,v in  labels_to_counts.iteritems():
            if v > best_score:
                best_score = v
                best_label = k 

        labels[idx1] = best_label
        logging.info('best label for idx1({}) is {}'.format(idx1, best_label))
    
    #remap the labels into a contiguous range. First we find the mapping
    label_remap ={}
    for i in range(len(labels)):
        next_id = len(label_remap)
        if labels[i] not in label_remap:
            logging.info('label:{} in label_remap {}?'.format(labels[i], label_remap))
            label_remap[labels[i]] = next_id

    logging.info('---------------------------------------label_remap:{}'.format(label_remap))

    logging.info('---------------------------------------before remap, label is:{}'.format(labels))
    for i in range(len(labels)):
        labels[i] = label_remap[labels[i]]
    logging.info('---------------------------------------after remap, label is:{}'.format(labels))

    return labels
    


#----------------------------------------------------------------------------------------
def chinese_whispers(edges, num_iterations):
    logging.info('unordered edges: {}'.format(edges))
    oedges = convert_unordered_to_ordered(edges)
    logging.info('ordered edges: {}'.format(oedges))
    oedges.sort(key=lambda tup: tup[0])  # sorts in place by the 1st element
    logging.info('sorted ordered edges: {}'.format(oedges))
    return chinese_whispers_ordered(oedges, num_iterations)


def load_edges_from_file(edge_file):
    edges = []
    with open(edge_file,'rb') as ef:
        for line in ef:
            n1, n2, w = line.strip('\n').split()
            edges.append( (int(n1), int(n2), float(w)) )
    return edges


if __name__=='__main__':

    edges = load_edges_from_file('./graph.txt')
    logging.info('edges: {}'.format(edges))
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    nx.draw(G, with_labels=True)
    plt.show()

    labels = chinese_whispers(edges, 3)
    logging.info('labels: {}'.format(labels))
    
    color_map = []
    for label in labels:
        if label == 0:
            color_map.append('yellow')
        else:
            color_map.append('red')
    nx.draw(G,node_color=color_map, with_labels=True)
    plt.show()





#initialize the graph

#G = nx.Graph()
#
#G.add_nodes_from(nodes)
#
#for n, v in enumerate(nodes):
#    print n, v
#    G.node[n]['class'] = n
#
#G.add_edges_from(edges) 
#
#nx.draw(G)
#plt.show()
#
#print 'edges type:', type(G.edges())
#
#
##run Chinese Whisper
#iterations = 10
#
#for z in range(0, iterations):
#    gn = G.nodes()
#    random.shuffle(gn)
#    for node in gn:
#        neighs = G[node]
#        print 'node:', node, ',neighs:', neighs
#        classes = {}
#        #do an inventory of the given nodes and edge weights
#        for ne in neighs:
#            if isinstance(ne, int):
#                if G.node[ne]['class'] in classes:
#                    classes[G.node[ne]['class']] += G[node][ne]['weight']
#                else:
#                    classes[G.node[ne]['class']] = G[node][ne]['weight']
#        #find the class with the highest edge weight sum
#        max_n = 0
#        maxclass = 0
#        for c in classes:
#            if classes[c] > max_n:
#                max_n = classes[c]
#                maxclass = c
#        #set the class of target node to the winning local class
#        G.node[node]['class'] = maxclass

                    
