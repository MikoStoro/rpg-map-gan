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
    generator.add(layers.Dense(3*3*128, use_bias=False, input_shape=(NOISE_DIMENSION,), kernel_initializer=weight_init))
    generator.add(layers.BatchNormalization())
    generator.add(layers.LeakyReLU())
    # Reshape 1D Tensor into 3D
    generator.add(layers.Reshape((3, 3, 128)))
    # First upsampling block
    generator.add(layers.Conv2DTranspose(128, kernel_size=3, strides=3, padding='same', use_bias=False, kernel_initializer=weight_init))
    generator.add(layers.BatchNormalization())
    generator.add(layers.LeakyReLU())
    generator.add(layers.Conv2D(1, kernel_size=3, padding='same', use_bias=False, kernel_initializer=weight_init))
    #generator.add(layers.Discretization([0.5]))
    rename_layers(generator, name)

    print(generator.summary())

    return generator
    