from defines import *
from gan_utils import *
import tensorflow
import numpy as np
from tensorflow.keras import backend as backend
import generator_model
import discriminator_model

import generator_model_9x9
import discriminator_model_9x9

DATASET = "datasets/dataset/dataset9x9.npy"
IMG_DIMENSION = 9
BATCH_SIZE = 1000

generator_optimizer = tensorflow.keras.optimizers.RMSprop(OPTIMIZER_LR)
discriminator_optimizer = tensorflow.keras.optimizers.RMSprop(OPTIMIZER_LR)

def print_training_progress(batch, generator_loss, discriminator_loss):
  """ Print training progress. """
  print('Losses after mini-batch %5d: generator %e, discriminator %e' % (batch, generator_loss, discriminator_loss))


@tensorflow.function
def perform_train_step(real_images, generator, discriminator):
  """ Perform one training step with Gradient Tapes """
  
    ##train discriminator
  for i in range(DISC_TRAIN_RATIO):
    noise = generate_noise(BATCH_SIZE)
    with tensorflow.GradientTape() as discriminator_tape:
      generated_images = generator(noise, training=True)
      discriminated_generated_images = discriminator(generated_images, training=True)
      discriminated_real_images = discriminator(real_images, training=True)
      discriminator_loss = -(backend.mean(discriminated_real_images) - backend.mean(discriminated_generated_images)) 
    
    discriminator_gradients = discriminator_tape.gradient(discriminator_loss, discriminator.trainable_variables)  
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
    
  with tensorflow.GradientTape() as generator_tape:
  ##train generator
    noise = generate_noise(BATCH_SIZE)
    generated_images = generator(noise, training=True)
    discriminated_generated_images = discriminator(generated_images, training=True)
    generator_loss = -(backend.mean(discriminated_generated_images))

  generator_gradients = generator_tape.gradient(generator_loss, generator.trainable_variables)
  generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
      
  return (generator_loss, discriminator_loss)

def train_gan(num_epochs, image_data, generator, discriminator):
  """ Train the GAN """
  # Perform one training step per batch for every epoch
  for epoch_no in range(num_epochs):
    num_batches = image_data.__len__()
    print(f'Starting epoch {epoch_no+1} with {num_batches} batches...')
    batch_no = 0
    # Iterate over batches within epoch
    for batch in image_data:
      generator_loss, discriminator_loss = perform_train_step(batch, generator, discriminator)
      batch_no += 1
      DISCRIMINATOR_LOSS.append(discriminator_loss)
      GENERATOR_LOSS.append(generator_loss)
      # Print statistics and generate image after every n-th batch
      if batch_no % PRINT_STATS_AFTER_BATCH == 0:
        print_training_progress(batch_no, generator_loss, discriminator_loss)
        generate_image(generator, epoch_no, batch_no, IMG_DIMENSION, IMG_DIMENSION)
        visualize_loss(epoch_no, batch_no)
    

    # Save models on epoch completion.
    #####save_models(generator, discriminator, epoch_no)
  # Finished :-)
  print(f'Finished unique run {UNIQUE_RUN_ID}')


def load_data():
    maps = np.load(DATASET)
    maps = np.array(maps)
    maps = maps.astype('float16')
    dataset = tensorflow.data.Dataset.from_tensor_slices(maps).shuffle(10000).repeat(1).batch(BATCH_SIZE)
    return dataset
    
def run_gan():
  """ Initialization and training """
  # Make run directory
  make_directory_for_run()
  # Set random seed
  tensorflow.random.set_seed(42)
  # Get image data
  data = load_data()
  # Create generator and discriminator
  generator = generator_model_9x9.create()
  discriminator = discriminator_model_9x9.create(wasserstein=True)
  # Train the GAN
  print('Training GAN ...')
  train_gan(NUM_EPOCHS, data, generator, discriminator)

run_gan()
