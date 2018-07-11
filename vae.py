from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from util import *

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

from plotly.offline import plot
import plotly.graph_objs as go

def create_trace(x_set, encoder, name='set name'):
    z_mean, _, _ = encoder.predict(x_set, batch_size=128)
    return go.Scatter(
        x=z_mean[:,0], y=z_mean[:,1],
        mode='markers', name=name,
        marker=dict(color='rgb(0,0,0)')) 

def plot_results(models, x_test, x_test1, x_train, batch_size=128, model_name="vae_mlp", ids=[], test_ids=[]):
    encoder, decoder = models

    if not os.path.exists(model_name):
        os.mkdir(model_name)

    filename = os.path.join(model_name, "vae_mean.png")
    z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)

    #from sklearn.cluster import KMeans
    from sklearn.cluster import AgglomerativeClustering    

    #km = KMeans(n_clusters=4).fit(z_mean)
    #labels = km.labels_
    #centers = km.cluster_centers_

    ac = AgglomerativeClustering(n_clusters=4).fit(z_mean)
    labels = ac.labels_

    from sklearn.svm import SVC
    xs = z_mean.copy()

    num_clust = 4
    traces = []

    l, r = z_mean[:,0].min(), z_mean[:,0].max()

    colors = ['rgb(255,51,51)', 'rgb(51,153,51)', 'rgb(0,0,255)', 'rgb(255,153,0)']
    for i in range(num_clust):
        idxs = np.argwhere(labels == i).flatten()
        z_clust = z_mean[idxs]
        x_clust = x_test[idxs]

        if i <= num_clust-1:
            clf = SVC(kernel='linear')
            ys = np.zeros(len(xs))
            ys[idxs] = 1
            clf.fit(xs, ys)
            ms = clf.coef_[0]
            inter = clf.intercept_[0]

            m, b = -ms[0]/ms[1], -inter/ms[1]
            traces.append(go.Scatter(
                x=[l, r], y=[l*m+b, r*m+b],
                mode='lines', name='boundary'))

        traces.append(go.Scatter(x=z_clust[:,0], y=z_clust[:,1],
            mode='markers', name='cluster%d' % i,
            text=arr(ids)[idxs],
            marker=dict(
                opacity=0.5,
                color=colors[i])))

    '''
    traces.append(go.Scatter(
        x=centers[:,0], y=centers[:,1],
        mode='markers', name='center',
        marker=dict(
            color='rgb(0,0,0)',
            size=10)))
    '''
    if not x_train is None:
        traces.append(create_trace(x_train, encoder, name='train'))

    if not x_test1 is None:
        traces.append(create_trace(x_test1, encoder, name='stage 1'))

    layout = dict(
        title='Test Data',
        hovermode='closest',
        xaxis=dict(
            title='z0',
            range=[-8.5, 2.15]),
        yaxis=dict(
            title='z1',
            range=[-5.15, 2.15]))
    plot({'data': traces, 'layout': layout}, filename='vae.html')


original_dim = 128*128
input_shape = (original_dim, )
intermediate_dim = 512
batch_size = 128
latent_dim = 2
epochs = 50

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

from cv2 import cvtColor
from cv2 import COLOR_BGRA2GRAY as bgr2gray
def load_imgs(d):
    ids = os.listdir(d)
    imgs = [imread(join(d, id, 'images', id+'.png')) for id in ids]

    for i, x in enumerate(imgs):
        if len(x.shape) == 3:
            imgs[i] = cvtColor(x, bgr2gray)

    return imgs, ids

if __name__ == '__main__':
    #X, ids = all_imgs(ret_ids=True)
    #X, y, ids = get_test_data(ret_ids=True)
    #X_tmp, test_ids = load_imgs('test_data2')
    X, ids = load_imgs('test_data2')
    test_ids = ids
    #X.extend(X_tmp)
    #ids.extend(test_ids)
    #test_ids = []
    X_test1 = get_test_data(just_X=True)
    X_train = all_imgs()

    from scipy.misc import imresize
    X = arr([imresize(x, (128, 128))/255. for x in X])
    X = X.reshape(len(X), -1)

    X_test1 = arr([imresize(x, (128, 128))/255. for x in X_test1])
    X_test1 = X_test1.reshape(len(X_test1), -1)

    X_train = arr([imresize(x, (128, 128))/255. for x in X_train])
    X_train = X_train.reshape(len(X_train), -1)

    #m = 0 
    #m = int(len(X)*.9)
    #x_train = X[:m]
    #x_test = X[m:]

    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    args = parser.parse_args()
    models = (encoder, decoder)

    reconstruction_loss = mse(inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    if args.weights:
        early_stop = EarlyStopping(patience=3, verbose=1)
        checkpoint = ModelCheckpoint('vae_test2.h5', verbose=1, save_best_only=True)
        vae = vae.load_weights(args.weights)
    else:
        vae.fit(x_train,
                epochs=50,
                batch_size=batch_size,
                validation_data=(x_test, None))
        vae.save_weights('vae_test2.h5')

    plot_results(models,
                 X, X_test1, X_train,
                 batch_size=batch_size,
                 model_name="vae_test2",
                 ids=ids, test_ids=test_ids)
