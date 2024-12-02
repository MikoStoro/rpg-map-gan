import tensorflow as tf
from gan_utils import *
import sys
sys.path.insert(1, 'data_utils')
#import data_utils.dataSetCreator as dataCreator
import input_generator as input_generator
import pix_disc_model_256 as disc
import pix_gen_model_256 as gen
import time
import pickle
from pathlib import Path



DATASET_NAME = "datasets/caves"
LAMBDA = 100 #100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

input_dataset_train = None
with open("./" + DATASET_NAME +"_inputs_train", "rb") as f:
  input_dataset_train = pickle.load(f)

target_dataset_train = None
with open("./" + DATASET_NAME +"_targets_train", "rb") as f:
  target_dataset_train = pickle.load(f)

input_dataset_test = None
with open("./" + DATASET_NAME +"_inputs_test", "rb") as f:
  input_dataset_test = pickle.load(f)

target_dataset_test = None
with open("./" + DATASET_NAME +"_targets_test", "rb") as f:
  target_dataset_test = pickle.load(f)

dataset_train = tf.data.Dataset.from_tensor_slices((input_dataset_train, target_dataset_train))
dataset_test = tf.data.Dataset.from_tensor_slices((input_dataset_test, target_dataset_test))
IMG_DIMENSION = 9
BATCH_SIZE = 1
run_name = "caves_100_1"
output_dir = "./" + run_name + "/outputs"
log_dir = "./" + run_name + "/logs"
models_dir = "./" + run_name + "/models"



Path(output_dir).mkdir(parents=True, exist_ok=True)
Path(log_dir).mkdir(parents=True, exist_ok=True)
Path(models_dir).mkdir(parents=True, exist_ok=True)

generator = gen.create()
discriminator = disc.create()

generator.summary()
discriminator.summary()

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator=generator,
                                discriminator=discriminator)


def generate_images(model, test_input, tar, display = False):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  if display:
    plt.show()
  else:
    filename = output_dir + "/training_output_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".png"
    plt.savefig(filename)
  plt.close()

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = ((101-LAMBDA) * gan_loss) + (LAMBDA * l1_loss)

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

def normalize(img):
    img = (img / 127.5) - 1
    return img

def fit(dataset : tf.data.Dataset, dataset_test : tf.data.Dataset, steps):

  start = time.time()

  for step, (input_image, target) in dataset.shuffle(buffer_size=10).repeat().batch(1).take(steps).enumerate():
    
    #random rotation
    k = np.random.randint(0,4)
    input_image = tf.image.rot90(input_image, k)
    target = tf.image.rot90(target, k)
    
    # Random mirroring
    if tf.random.uniform(()) > 0.5:
      input_image = tf.image.flip_left_right(input_image)
      target = tf.image.flip_left_right(target)
    
    if tf.random.uniform(()) > 0.5:
      input_image = tf.image.flip_up_down(input_image)
      target = tf.image.flip_up_down(target)

    if (step) % 1000 == 0:
      #display.clear_output(wait=True)

      if step != 0:
        print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

      start = time.time()
      example_input, example_target = next(iter(dataset_test.shuffle(buffer_size=10, reshuffle_each_iteration=True).batch(1).take(1)))
      fake_input = next(iter(tf.data.Dataset.from_tensor_slices([normalize(input_generator.get_input(["stone", "none", "lava", "stone_manmade"]))]).batch(1).take(1)))
      generate_images(generator, fake_input, example_target)
      generate_images(generator, example_input, example_target)
      print(f"Step: {step//1000}k")

    train_step(input_image, target, step)

    # Training step
    if (step+1) % 10 == 0:
      print('.', end='', flush=True)


    # Save (checkpoint) the model every 2k steps
    if (step +1) % 2000 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)
      discriminator.save(models_dir + "/disc_model_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".keras")
      generator.save(models_dir + "/gen_model_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") +".keras")


fit(dataset_train, dataset_test, steps=40000)
