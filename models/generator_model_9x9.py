import keras.src.layers
import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import backend as backend
from tensorflow.keras.constraints import Constraint
from defines import *
from gan_utils import rename_layers


class RoundLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(RoundLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tensorflow.round(inputs)

    def get_config(self):
        config = super(RoundLayer, self).get_config()
        return config


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
    generator.add(layers.Conv2DTranspose(24, (3, 3), strides=(3, 3), padding='same', use_bias=False, kernel_initializer=weight_init))
    generator.add(layers.BatchNormalization())
    generator.add(layers.LeakyReLU())
    # Second upsampling block
    generator.add(layers.Conv2D(1, (3,3), strides=(1, 1), padding='same', use_bias=False, activation='linear', kernel_initializer=weight_init))

    # Layer translating to 0 or 1
    generator.add(layers.Dense(1))
    generator.add(layers.Lambda(lambda x: 100 * x))
    generator.add(layers.Activation('sigmoid'))
    #generator.add(layers.Discretization([0.5]))
    # rename_layers(generator, name)

    print(generator.summary())

    return generator
    