import networkx as nx
import numpy as np
from util import *

import matplotlib.pyplot as plt

def create_graph(img):
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
            if val <= cutoff:
                continue

            # Get the adjacent pixels of the current pixel.
            T, B = max(0, i-1), min(img.shape[0], i+2)
            L, R = max(0, j-1), min(img.shape[1], j+2)
            neighbors = img[T:B, L:R]

            # If there is a neighboring pixel that is at least intvl+1 less than
            # the current value, then consider any value at least 1 less than
            # the current value to be a "lower point".
            #
            # Otherwise, only consider pixels at least intvl+1 lower to be "lower points".
            intvl = 2
            if len(np.argwhere(neighbors < val-intvl)) > 0:
                intvl = 0

            lower_pts = np.argwhere(neighbors < val-intvl)
            for pt in lower_pts:
                p = (pt[0]+T, pt[1]+L)
                lowval = img[p[0], p[1]]
                weight = 1. - (maxval-lowval)/float(maxval)
                dg.add_weighted_edges_from([((j, i), (p[1], p[0]), weight)])

            # If there are no neighboring pixels with lower values, don't add
            # edges to equal neighboring pixels. This will just accentuate noise.
            if len(lower_pts) == 0:
                continue

            eq_pts = np.argwhere((neighbors <= val+intvl) & (neighbors >= val-intvl))
            for pt in eq_pts:
                p = (pt[0]+T, pt[1]+L)

                # Don't add self-edges
                if p[0] == i and p[1] == j:
                    continue

                # If the pixel has neighbors that are less than it, add two edges
                # between any neighboring pixel that is equal to it
                dg.add_weighted_edges_from([
                    ((j, i), (p[1], p[0]), 0),
                    ((p[1], p[0]), (j, i), 0)])
    return dg

def get_centroids_from_graph(sources, ret_clusters=False):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN

    X = StandardScaler().fit_transform(sources.astype(np.float64))
    db = DBSCAN(eps=0.1, min_samples=3).fit(X)
    labels = db.labels_
    num_cluster = len(set(labels)) - (1 if -1 in labels else 0)

    for label_num in range(num_cluster):
        pts = sources[np.argwhere(labels == label_num).flatten()]
        cm = np.mean(pts, axis=0)
        if ret_clusters:
            yield cm, pts
        else:
            yield cm
    

def find_sources(img=None, plot=True, pct=99.5):
    if img is None:
        img, mask = get_img(denoise=False, ret_mask=True)
    dg = create_graph(img)
    
    # Create a dictionary mapping each node to its out-degree
    sources = {n: dg.out_degree(n, weight='weight') for n in dg.nodes if dg.in_degree(n, weight='weight') == 0}
    source_nodes = arr([k for (k, v) in sources.iteritems()])
    degrees = arr([v for (k, v) in sources.iteritems()])
    
    # Argsort the list of degrees in descending order
    idxs = list(reversed(np.argsort(degrees).tolist()))

    # Get the indeces of the nodes in the pct-th percentile of out-degree
    def sources_at_pct(pct, prev_count=0):
        deg = np.percentile(degrees, pct)
        count = np.sum(degrees > deg)
        return source_nodes[idxs[prev_count:count]], count

    if plot:
        _, ax = plt.subplots(1, 2)
        gray_imshow(ax[0], img)

    sources, start = sources_at_pct(pct)
    for cm, pts in get_centroids_from_graph(sources, ret_clusters=True):
        if plot:
            #ax[0].scatter(pts[:,0], pts[:,1], s=10)
            ax[0].scatter(cm[0], cm[1], s=50, marker='*')
        else:
            yield cm, pts

    if plot:
        gray_imshow(ax[1], mask)
        plt.show()

def find_scc():
    img, mask = get_img(ret_mask=True)
    dg = create_graph(img)

    img_graph = np.zeros_like(img)
    for scc in nx.strongly_connected_component_subgraphs(dg):
        if len(scc.nodes) <= 2:
            continue

        for node in scc.nodes:
            img_graph[node[1], node[0]] = 1

    _, ax = plt.subplots(1, 3)
    gray_imshow(ax[0], img, title='Original')
    gray_imshow(ax[1], img_graph, title='SCC Graph')
    gray_imshow(ax[2], mask, title='Mask')
    plt.show()

if __name__ == '__main__':
    find_sources() 
    #find_scc()
