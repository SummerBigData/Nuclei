from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from util import *

from keras.layers import Lambda, Input, Dense, Conv2D, Deconv2D, Flatten, Reshape
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
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]

    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

from plotly.offline import plot
import plotly.graph_objs as go

def create_trace(x_set, encoder, name='set name', text=None):
    z_mean, _, _ = encoder.predict(x_set, batch_size=128)
    return go.Scatter(
        x=z_mean[:,0], y=z_mean[:,1],
        mode='markers', name=name, text=arr([id+':'+name for id in text]),
        marker=dict(opacity=0.5))
        #marker=dict(color='rgb(0,0,0)')) 

def plot_results(models, data, batch_size=128, model_name="cvae"):
    encoder, decoder = models

    if not os.path.exists(model_name):
        os.mkdir(model_name)

    filename = os.path.join(model_name, "cvae_mean.png")

    traces = []
    for x_set, id_set, name in data:
        traces.append(create_trace(x_set, encoder, name=name, text=id_set))

    layout = dict(
        title='Test Data',
        hovermode='closest',
        xaxis=dict(title='z0'),
        yaxis=dict(title='z1'))
    plot({'data': traces, 'layout': layout}, filename='cvae.html')


dim = 128
intermediate_dim = 512
input_shape = (dim, dim, 1)

cp1 = {'padding': 'same', 'activation': 'relu'}
cp2 = {'padding': 'same', 'activation': 'relu', 'strides': (2, 2)}
cp3 = {'padding': 'same', 'activation': 'relu', 'strides': (1, 1)}
cp4 = cp3
num_filter = 32

batch_size = 128
latent_dim = 2
epochs = 50

# VAE model = encoder + decoder
inputs = Input(shape=input_shape, name='encoder_input')
c1 = Conv2D(1, (2, 2), **cp1)(inputs)
c2 = Conv2D(num_filter, (3, 3), **cp2)(c1)
c3 = Conv2D(num_filter, (3, 3), **cp3)(c2)
c4 = Conv2D(num_filter, (3, 3), **cp4)(c3)
fl = Flatten()(c4)
x = Dense(intermediate_dim, activation='relu')(fl)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
h = Dense((dim//2)**2 * num_filter, activation='relu')(x)
re = Reshape((dim//2, dim//2, num_filter))(h)
dc1 = Deconv2D(num_filter, (3, 3), **cp4)(re)
dc2 = Deconv2D(num_filter, (3, 3), **cp3)(dc1)
dc3 = Deconv2D(1, (3, 3), **cp2)(dc2)
outputs = Deconv2D(1, (2, 2), padding='same', activation='sigmoid')(dc3)

decoder = Model(latent_inputs, outputs, name='decoder')
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='cvae')

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
    X, ids = load_imgs('test_data2')
    X_test1, test1_ids = get_test_data(just_X=True, ret_ids=True)
    X_train, train_ids = all_imgs(ret_ids=True)

    from scipy.misc import imresize
    X = arr([imresize(x, (128, 128))/255. for x in X])
    X_test1 = arr([imresize(x, (128, 128))/255. for x in X_test1])
    X_train = arr([imresize(x, (128, 128))/255. for x in X_train])

    X = np.expand_dims(X, axis=3)
    X_test1 = np.expand_dims(X_test1, axis=3)
    X_train = np.expand_dims(X_train, axis=3)

    #m = len(X)
    #while m % batch_size > 0:
    #    m -= 1
    #X = X[:m]

    m = len(X) - 2*batch_size
    x_train = X[:m]
    x_test = X[m:]
    print(x_train.shape, x_test.shape)

    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    args = parser.parse_args()
    models = (encoder, decoder)

    reconstruction_loss = mse(inputs, outputs)
    reconstruction_loss *= dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    if args.weights:
        vae = vae.load_weights(args.weights)
    else:
        early_stop = EarlyStopping(patience=3, verbose=1)
        checkpoint = ModelCheckpoint('cvae_test2.h5', verbose=1, save_best_only=True)
        vae.fit(x_train,
                epochs=35,
                batch_size=batch_size,
                validation_data=(x_test, None),
                callbacks=[early_stop, checkpoint])
        vae.save_weights('cvae_test2.h5')

    import matplotlib.pyplot as plt
    x = X[np.random.randint(0, len(X))]
    plt.imshow(x[:,:,0], 'gray')
    plt.show()

    z_mean = encoder.predict(np.expand_dims(x, axis=0))[0]
    print(z_mean)

    p = decoder.predict(z_mean)
    print(p.shape)
    plt.imshow(p[0,:,:,0], 'gray')
    plt.show()
    exit()

    data = [
        (X, ids, 'stage 2'),
        (X_train, train_ids, 'train'),
        (X_test1, test1_ids, 'stage 1')
    ]
    plot_results(models, data,
                 batch_size=batch_size,
                 model_name="cvae_test2")
