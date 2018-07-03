from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from util import *
from scipy.misc import imresize

imgs = arr([imresize(img, (256, 256))/255. for img in all_imgs(white=True)])
imgs = imgs.reshape(len(imgs), -1)

X = StandardScaler().fit_transform(imgs)
db = DBSCAN(eps=150., min_samples=5).fit(X)
labels = db.labels_
num_cluster = len(set(labels)) - (1 if -1 in labels else 0)
print num_cluster

import matplotlib.pyplot as plt
for i in range(num_cluster):
    cluster = imgs[np.argwhere(labels == i).flatten()]
    print len(cluster)
    for img in cluster:
        plt.imshow(img.reshape(256, 256), 'gray')
        plt.show()
