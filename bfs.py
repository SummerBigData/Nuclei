from util import *
from graph import find_sources 
from networkx import bfs_edges
from Queue import Queue

def traverse(img, src, visited):
    global ax_vis
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

        intvl = 1
        if len(np.argwhere(neighbors < curr_val-intvl)) > 0:
            intvl = 0

        to_visit = np.argwhere((neighbors<=curr_val-intvl) & (vis == 0) & (neighbors>=5))
        for pt in to_visit:
            pt = (pt[0]+T, pt[1]+L)
            nodes.append([pt[0], pt[1]])
            q.put(pt)
            visited[pt[0], pt[1]] = 1
    
            ax_vis.set_data(visited)
            plt.pause(0.0005)

    return arr(nodes), visited


import matplotlib.pyplot as plt
img, mask = get_img(ret_mask=True)
model_b, model_w = load_unets('unet-uneroded', name_white='unet-white2')

batch = batchify(img/255., unet=False)

if np.mean(img) >= 127.5:
    p = model_w.predict(batch)
else:
    p = model_b.predict(batch)
p = unbatchify(p)

#from scipy.misc import imresize
#p = imresize(p, img.shape)

_, ax = plt.subplots(1, 3)
gray_imshow(ax[0], img, title='Original')
gray_imshow(ax[1], mask, title='Mask')

p = (p*255).astype(np.uint8)
visited = np.zeros_like(p)
visited[0,0] = 1
ax_vis = gray_imshow(ax[2], visited, title='Visited')

def start_bfs(e):
    global p, visited
    print 'Finding sources'

    for cm, _ in find_sources(img=p, plot=False, pct=95.):
        ax[0].scatter(cm[0], cm[1], s=50, marker='*')

        pts, visited = traverse(img, cm, visited)
        #if len(pts) > 0:
        #    ax[0].scatter(pts[:,1], pts[:,0])

    print 'BFS is complete'

from matplotlib.widgets import Button
ax_btn = plt.axes([0.15, 0.915, 0.15, 0.03])
btn = Button(ax_btn, 'Start BFS')
btn.on_clicked(start_bfs)

plt.show()
