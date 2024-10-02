import tensorflow as tf
from gan_utils import *
LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

DATASET = "datasets/dataset/dataset9x9.npy"
IMG_DIMENSION = 9
BATCH_SIZE = 1000


def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

@tf.function
def train_step(input_image, target, step):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    
  generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
  
  generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
    tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

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
  # Get image data
  data = load_data()
  # Create generator and discriminator
  generator = generator_model_9x9.create()
  discriminator = discriminator_model_9x9.create(wasserstein=True)
  # Train the GAN
  print('Training GAN ...')
  train_gan(NUM_EPOCHS, data, generator, discriminator)

run_gan()