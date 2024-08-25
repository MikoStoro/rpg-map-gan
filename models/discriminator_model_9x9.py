import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import backend as backend
from tensorflow.keras.constraints import Constraint
from defines import *


class ClipConstraint(Constraint):
 # set clip value when initialized
 def __init__(self, clip_value):
  self.clip_value = clip_value
  
  # clip model weights to hypercube
  def __call__(self, weights):
   return backend.clip(weights, -self.clip_value, self.clip_value)
 
 # get the config
 def get_config(self):
  return {'clip_value': self.clip_value}


    
def create(wasserstein = True, name="disc"):
    discriminator = tensorflow.keras.Sequential()
    
    weight_init = tensorflow.keras.initializers.RandomNormal(stddev=WEIGHT_INIT_STDDEV, seed = 42)
    constraint = ClipConstraint(CLIPPING)
    activation = 'linear'
    
    
    
    if not wasserstein:
        activation = 'tanh'
    
    # First Convolutional block
    discriminator.add(layers.Input(shape=(9,9,1)))
    discriminator.add(layers.Conv2D(24, (3, 3), strides=(3, 3), padding='same', kernel_initializer=weight_init, kernel_constraint=constraint))
    discriminator.add(layers.LeakyReLU())
    discriminator.add(layers.Dropout(0.25))
    # Second Convolutional block
    discriminator.add(layers.Conv2D(12, (2, 2), strides=(3, 3), padding='same', kernel_initializer=weight_init,  kernel_constraint=constraint))
    discriminator.add(layers.LeakyReLU())
    discriminator.add(layers.Dropout(0.25))
    # Flatten and generate output prediction
    discriminator.add(layers.Flatten())
    discriminator.add(layers.Dense(1, kernel_initializer=weight_init, activation=activation))

    try: 
        for layer in discriminator.layers:
            newname = name + "_" + layer._name
            layer._name = newname
    except: pass
    print(discriminator.summary())

    return discriminator