import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from util import *
from graph import find_sources 
from networkx import bfs_edges
from Queue import Queue

def traverse(img, src, visited):
    src = (int(src[1]), int(src[0]))
    #graph = create_graph(img)
    #edges = list(bfs_edges(graph, (int(src[0]), int(src[1]))))
    #nodes = set([node for edge in edges for node in edge])
    nodes = []
    visited[src[0], src[1]] = 1

    q = Queue()
    q.put(src)
    while not q.empty():
        i, j = q.get()
        curr_val = img[i, j]

        T, B = max(0, i-1), min(img.shape[0], i+2)
        L, R = max(0, j-1), min(img.shape[1], j+2)
        neighbors = img[T:B, L:R]
        vis = visited[T:B, L:R]

        to_visit = np.argwhere((neighbors<=curr_val) & (vis == 0) & (neighbors>=5))
        for pt in to_visit:
            pt = (pt[0]+T, pt[1]+L)
            nodes.append([pt[0], pt[1]])
            q.put(pt)
            visited[pt[0], pt[1]] = 1

    return arr(nodes), visited

img, mask = get_img(ret_mask=True)
model_b, model_w = load_unets('unet-uneroded', name_white='unet-white2')

batch = batchify(img/255.)
if np.mean(img) >= 127.5:
    p = model_w.predict(batch)
else:
    p = model_b.predict(batch)
p = unbatchify(p)

import matplotlib.pyplot as plt
_, ax = plt.subplots(1, 3)
gray_imshow(ax[0], img, title='Original')
gray_imshow(ax[1], mask, title='Mask')
gray_imshow(ax[2], p, title='predicted')

p = (p*255).astype(np.uint8)
visited = np.zeros_like(p)

for cm, _ in find_sources(img=p, plot=False, pct=95.):
    ax[0].scatter(cm[0], cm[1], s=50, marker='*')

    pts, visited = traverse(img, cm, visited)
    if len(pts) > 0:
        ax[0].scatter(pts[:,0], pts[:,1])

#for centr in centroids_from_img(p):
#    ax[0].scatter(centr[1], centr[0], s=50, marker='*') 

plt.show()
