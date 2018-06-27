import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from cv2 import fastNlMeansDenoising as mean_denoise
from cv2 import erode, dilate
from scipy.ndimage import binary_fill_holes as fill_holes

from util import *

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
    img, mask = get_img(denoise=False, ret_mask=True)

    """
    # Increase contrast
    LUT = np.zeros(256, dtype=np.uint8)
    LUT[img.min():img.max()+1] = np.linspace(0, 255, num=(img.max()-img.min())+1, endpoint=True, dtype=np.uint8)
    img = LUT[img]
    """

    maxval = img.max() 

    dg = nx.grid_2d_graph(img.shape[0], img.shape[1]).to_directed()
    ebunch = [e for e in dg.edges]
    dg.remove_edges_from(ebunch)

    mult = 1.0
    if np.mean(img) >= 127.5:
        mult = -1.0
    cutoff = int(np.mean(img)/2)

    for i, row in enumerate(img):
        for j, val in enumerate(row):
            #if val == 0:
            if val <= cutoff:
                continue

            T, B = max(0, i-1), min(img.shape[0], i+2)
            L, R = max(0, j-1), min(img.shape[1], j+2)
            neighbors = img[T:B, L:R]

            lower_pts = np.argwhere(neighbors < val-1)
            for pt in lower_pts:
                p = (pt[0]+T, pt[1]+L)
                lowval = img[p[0], p[1]]
                weight = 1. - (maxval-lowval)/float(maxval)
                dg.add_weighted_edges_from([((j, i), (p[1], p[0]), weight)])

            #if len(lower_pts) == 0:
            #    continue

            #eq_pts = np.argwhere(neighbors == val)
            eq_pts = np.argwhere((neighbors <= val+1) & (neighbors >= val-1))
            for pt in eq_pts:
                p = (pt[0]+T, pt[1]+L)
                dg.add_weighted_edges_from([
                    ((j, i), (p[1], p[0]), 0),
                    ((p[1], p[0]), (j, i), 0)])
    
    sources = {n: mult*dg.out_degree(n, weight='weight') for n in dg.nodes}
    source_nodes = arr([k for (k, v) in sources.iteritems()])
    degrees = arr([v for (k, v) in sources.iteritems()])
    
    #plt.hist(degrees)
    #plt.show()

    idxs = list(reversed(np.argsort(degrees).tolist()))

    #mean_degree = np.mean(degrees)
    mean_degree = np.percentile(degrees, 99.5)
    count_above_mean = np.sum(degrees > mean_degree)
    print count_above_mean, len(source_nodes)
    sources = source_nodes[idxs[:count_above_mean]]

    """
    img_graph = np.zeros_like(img)
    for scc in nx.strongly_connected_component_subgraphs(dg):
        if len(scc.nodes) <= 5:
            continue

        for node in scc.nodes:
            img_graph[node[1], node[0]] = 1
    """


    #from skimage.measure import find_contours
    #contours = find_contours(img_graph, 0.0)
    _, ax = plt.subplots(1, 2)
    gray_imshow(ax[0], img)
    #gray_imshow(ax[1], img_graph)
    ax[0].scatter(sources[:,0], sources[:,1], s=10, c='r')
    #for c in contours:
    #    if len(c) <= 15:
    #        continue
    #    ax[1].plot(c[:,1], c[:,0])
    gray_imshow(ax[1], mask)
    plt.show()

find_sources() 

def disp_img_connectivity(k=5):
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
