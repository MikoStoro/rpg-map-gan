import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import backend as backend
from tensorflow.keras.constraints import Constraint
from defines import *
from gan_utils import rename_layers

    


def create(name = "gen"):
    weight_init = tensorflow.keras.initializers.RandomNormal(stddev=WEIGHT_INIT_STDDEV, seed = 42)

    
    """ Create Generator """
    generator = tensorflow.keras.Sequential()
    # Input block
    generator.add(layers.Dense(4*4*128, use_bias=False, input_shape=(NOISE_DIMENSION,), kernel_initializer=weight_init))
    generator.add(layers.BatchNormalization())
    generator.add(layers.LeakyReLU())
    # Reshape 1D Tensor into 3D
    generator.add(layers.Reshape((4, 4, 128)))
    # First upsampling block
    generator.add(layers.Conv2DTranspose(22, (6, 6), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=weight_init))
    generator.add(layers.BatchNormalization())
    generator.add(layers.LeakyReLU())
    # Second upsampling block
    generator.add(layers.Conv2DTranspose(66, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=weight_init))
    generator.add(layers.BatchNormalization())
    generator.add(layers.LeakyReLU())
    generator.add(layers.Conv2D(1, (3,3), strides=(1, 1), padding='same', use_bias=False, activation='linear', kernel_initializer=weight_init))

    #generator.add(layers.Discretization([0.5]))
    # rename_layers(generator, name)

    print(generator.summary())

    return generator
    