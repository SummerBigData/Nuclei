import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from util import *

k = 5 

def neighborhood(patch, i, j):
    neighbors = np.zeros_like(patch, dtype=np.float64)
    neighbors += np.inf
    t = max(0, i-1)
    b = min(patch.shape[0], i+2)
    l = max(0, j-1)
    r = min(patch.shape[1], j+2)
    neighbors[t:b, l:r] = patch[t:b, l:r]
    return neighbors

def find_sources():
    img = get_img()
    maxval = img.max() 

    dg = nx.grid_2d_graph(k, k).to_directed()
    ebunch = [e for e in dg.edges]
    dg.remove_edges_from(ebunch)

    for i, row in enumerate(img):
        for j, val in enumerate(row):
            if val == 0:
                continue

            lower_pts = np.argwhere(neighborhood(img, i, j) < val)
            for pt in lower_pts:
                lowval = img[pt[0], pt[1]]
                dg.add_weighted_edges_from([((j, i), (pt[1], pt[0]), 1)])

            eq_pts = np.argwhere(neighborhood(img, i, j) == val)
            for pt in eq_pts:
                dg.add_weighted_edges_from([
                    ((j, i), (pt[1], pt[0]), 1),
                    ((pt[1], pt[0]), (j, i), 1)])
    
    #sources = {n: dg.out_degree(n, weight='weight') for n in dg.nodes}
    #sources = dict(sorted(sources.iteritems(), key=lambda (k,v): (v,k)))
    #print sources
    for scc in nx.strong_connected_components_subgraphs(dg):
        for node in scc.nodes:
            img[node[0], node[1]] = 255
    img[img<255] = 0

    gray_imshow(plt, img)
    plt.show()

find_sources() 

def disp_img_connectivity():
    img = get_img()

    y, x = np.random.choice(range(img.shape[0]-k)), np.random.choice(range(img.shape[1]-k))
    patch = img[y:y+k, x:x+k]
    maxval = patch.max()

    dg = nx.grid_2d_graph(k, k).to_directed()
    ebunch = [e for e in dg.edges]
    dg.remove_edges_from(ebunch)
    pos = dict((n, (n[0], k-n[1])) for n in dg.nodes())

    d = 2.5
    for i, row in enumerate(patch):
        for j, val in enumerate(row):
            if val == 0:
                continue

            lower_pts = np.argwhere(neighborhood(patch, i, j) < val)
            for pt in lower_pts:
                lowval = patch[pt[0], pt[1]]
                weight = d * (1 - (maxval-lowval)/float(maxval))
                dg.add_weighted_edges_from([((j, i), (pt[1], pt[0]), weight)])

            eq_pts = np.argwhere(neighborhood(patch, i, j) == val)
            for pt in eq_pts:
                dg.add_weighted_edges_from([
                    ((j, i), (pt[1], pt[0]), d),
                    ((pt[1], pt[0]), (j, i), d)])

    _, axs = plt.subplots(1, 2)
    axs[1].axis('off')
    nx.draw_networkx_nodes(dg, pos, node_size=350)

    all_weights = []
    for (n1, n2, data) in dg.edges(data=True):
        all_weights.append(data['weight'])

    for w in list(set(all_weights)):
        w_edges = [(n1, n2) for (n1, n2, ea) in dg.edges(data=True) if ea['weight'] == w]
        nx.draw_networkx_edges(dg, pos, edgelist=w_edges, width=w)

    nx.draw_networkx_labels(dg, pos, font_size=10)

    for i in range(k):
        for j in range(k):
            color = 'k'
            if patch[j, i] < patch.mean():
                color = 'w'
            axs[0].text(i, j, patch[j, i], 
                        ha='center', va='center',
                        color=color)

    gray_imshow(axs[0], patch)
    plt.show()

#for _ in range(5):
#    disp_img_connectivity()
