import datetime
import tensorflow
import matplotlib.pyplot as plt

import os
import numpy as np
from defines import *

UNIQUE_RUN_ID = ""

def generate_run_name():
  global UNIQUE_RUN_ID
  now = datetime.datetime.now()
  UNIQUE_RUN_ID = now.strftime("%d-%m-%Y %H:%M:%S")  # dd/mm/YY H:M:S
  return UNIQUE_RUN_ID


def make_directory_for_run():
  """ Make a directory for this training run. """
  run_id = generate_run_name()
  print(f'Preparing training run {run_id}')
  if not os.path.exists('./runs'):
    os.mkdir('./runs')
  os.mkdir(f'./runs/{run_id}')
  
  
def generate_image(generator, epoch = 0, batch = 0,x = 16,y = 16):
  """ Generate subplots with generated examples. """
  images = []
  noise = generate_noise(BATCH_SIZE)
  images = generator(noise, training=False)
  plt.figure(figsize=(10.0, 10.0))
  index = 0
  for i in range(0,BATCH_SIZE, int(BATCH_SIZE/16)+1):
    # Get image and reshape
    image = images[i]
    image = np.reshape(image, (x, y))
    # Plot
    plt.subplot(4, 4, index+1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    index += 1
  if not os.path.exists(f'./runs/{UNIQUE_RUN_ID}/images'):
    os.mkdir(f'./runs/{UNIQUE_RUN_ID}/images')
  plt.savefig(f'./runs/{UNIQUE_RUN_ID}/images/epoch{epoch}_batch{batch}.jpg')
  plt.close()
  
def generate_data_image(data, x = 66, y = 66):
  """ Generate subplots with generated examples. """
  images = data
  plt.figure(figsize=(10.0, 10.0))
  index = 0
  for i in range(0,BATCH_SIZE, int(BATCH_SIZE/16)+1):
    # Get image and reshape
    image = images[i]
    image = np.reshape(image, (x, y))
    # Plot
    plt.subplot(4, 4, index+1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    index += 1
  if not os.path.exists(f'./runs/{UNIQUE_RUN_ID}/data'):
    os.mkdir(f'./runs/{UNIQUE_RUN_ID}/data')
  plt.savefig(f'./runs/{UNIQUE_RUN_ID}/data/data.jpg')
  plt.close()

def generate_noise(number_of_images = 1, noise_dimension = NOISE_DIMENSION):
  """ Generate noise for number_of_images images, with a specific noise_dimension """
  return tensorflow.random.normal([number_of_images, noise_dimension])

def save_models(generator, discriminator, epoch):
  """ Save models at specific point in time. """
  tensorflow.keras.models.save_model(
    generator,
    f'./runs/{UNIQUE_RUN_ID}/generator_{epoch}.model',
    overwrite=True,
    include_optimizer=True,
    save_format=None,
  )
  tensorflow.keras.models.save_model(
    discriminator,
    f'./runs/{UNIQUE_RUN_ID}/discriminator{epoch}.model',
    overwrite=True,
    include_optimizer=True,
    save_format=None, 
  )

def rename_layers(model, name):
  try: 
      for layer in model.layers:
          newname = name + "_" + layer._name
          layer._name = newname
  except:
      try: 
          for layer in model.layers:
              newname = name + "_" + layer._name
              layer.name = newname
      except: print("Could not set names for layers of model: " + name)
