from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from keras.layers import Dropout
from keras.layers.merge import concatenate

def unet_model(dropout=0.0, inpt_shape=None):
    if inpt_shape is None:
        inpt_shape = (None, None, 1)
    inputs = Input(inpt_shape)

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (inputs)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)
    d3 = Dropout(dropout) (p3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (d3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    d4 = Dropout(dropout) (p4)

    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (d4)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)
    d6 = Dropout(dropout) (c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (d6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)
    d7 = Dropout(dropout) (c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (d7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    return Model(inputs=[inputs], outputs=[outputs])
