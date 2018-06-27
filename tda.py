import kmapper as km
import numpy as np
import pandas as pd
from skimage.morphology import label

from util import *

imgs, ids = all_imgs(ret_ids=True)

# 1 column for total mean value
# 4 columns for the mean value in each quadrant
# 5 columns for the number of components when thresholded @ 10,20,30,40,50
data = np.zeros((len(imgs), 1+4+5))

data[:,0] = arr([np.mean(img) for img in imgs])
col = 1
for i in range(2):
    for j in range(2):
        data[:,col] = arr([
            np.mean(x[i*x.shape[0]//2:(i+1)*x.shape[0]//2, j*x.shape[1]//2:(j+1)*x.shape[1]//2])
            for x in imgs])
        col += 1

col = 5
for t in range(10, 60, 10):
    data[:,col] = [len(np.unique(label(threshold(x, t)))) for x in imgs]
    col += 1

import sklearn
mapper = km.KeplerMapper()
data_projected = mapper.fit_transform(data,
                                    projection=[0,1],
                                    #projection='knn_distance_5',
                                    scaler=sklearn.preprocessing.MinMaxScaler())

graph = mapper.map(data_projected,
                #inverse_X=data,
                nr_cubes=10,
                #perc_overlap=0.1,
                clusterer=sklearn.cluster.DBSCAN())

_ = mapper.visualize(graph,
                    path_html="tda_output.html",
                    inverse_X=data,
                    inverse_X_names=[
                        'Total Mean', 'Quad Mean 1', 'Quad Mean 2',
                        'Quad Mean 3', 'Quad Mean 4', '# Comp 10',
                        '# Comp 20', '# Comp 30', '# Comp 40', '# Comp 50'],
                    color_function=data[:,0])
