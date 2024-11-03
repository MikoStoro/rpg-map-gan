import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
DATA_PATH = "D:\\repos\\Github\\rpg-map-gan\\data_utils\\results\\"

def get_core_filenames():
    res = []
    for file in os.listdir(DATA_PATH):
        res.append(get_name_core(file))
    return list(set(res))

def normalize(img):
  img = (img / 127.5) - 1
  return img


def get_name_core(filename : str):
    filename = filename.replace("_ORIGINAL", "")
    filename = filename.replace("_COLORMAP", "")
    index = filename.rfind("_")
    filename = filename[:index]
    return filename
    
def extract_arrays(name_filter: None, limit: None):
    core_filenames = get_core_filenames()
    print(core_filenames)
    inputs = []
    targets = []

    for name in core_filenames:
        if name_filter is not None and name not in name_filter:
            continue
        index = 0
        while(True):
            input_name = name + "_COLORMAP_" + str(index) + ".npy"
            target_name = name + "_ORIGINAL_" + str(index) + ".npy"
            if not os.path.isfile(DATA_PATH + input_name) or not os.path.isfile(DATA_PATH + target_name):
                break
            input_img = normalize(tf.cast(np.load(DATA_PATH + input_name),tf.float32))
            target_img = normalize(tf.cast(np.load(DATA_PATH + target_name),tf.float32))
            inputs.append(input_img)
            targets.append(target_img)
            index += 1
            if limit is not None and index > limit:
                break
    return inputs, targets

def test_images(inputs, targets):
    for i in range(10):
        index = np.random.randint(0, len(inputs))
        plt.figure(figsize=(15, 15))
        
        display_list = [inputs[index], targets[index]]
        title = ['Input Image', 'Ground Truth']

        for i in range(2):
            plt.subplot(1, 2, i+1)
            plt.title(title[i])
            # Getting the pixel values in the [0, 1] range to plot.
            plt.imshow(display_list[i])
            plt.axis('off')
        plt.show()


#inputs, targets = extract_arrays(name_filter=["Fort_Joy_Exterior"], limit=100)

def create_dataset(inputs, targets):
    return tf.data.Dataset.from_tensor_slices(inputs, targets)

def save_dataset(dataset, path):
    dataset.save(path)
#test_images(inputs,targets)
inputs, targets = extract_arrays(name_filter=["Fort_Joy_Exterior"], limit = 100)
#np.save("./fort_joy_dataset", np.array((inputs,targets)))
np.save("./fort_joy_dataset_100_inputs", inputs)
np.save("./fort_joy_dataset_100_targets", targets)