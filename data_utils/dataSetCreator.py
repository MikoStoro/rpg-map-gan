import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
import random
import pickle
from sklearn.model_selection import train_test_split
DATA_PATH = "/home/mikostoro/Documents/GitHub/rpg-map-gan/overworld/"
DATASET_NAME = "overworld"

with tf.device("/cpu:0"):
    def get_core_filenames(path):
        res = []
        for file in os.listdir(path):
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
        
    def extract_arrays(name_filter =  None, limit = None, path = DATA_PATH):
        core_filenames = get_core_filenames(path)
        print(core_filenames)
        inputs = []
        targets = []

        for name in core_filenames:
            if name_filter is not None and name not in name_filter:
                continue
            index = 0
            this_map_input = []
            this_map_target = []
            while(True):
                input_name = name + "_COLORMAP_" + str(index) + ".npy"
                target_name = name + "_ORIGINAL_" + str(index) + ".npy"
                if not os.path.isfile(path + input_name) or not os.path.isfile(path + target_name):
                    print(os.path.isfile(path + input_name))
                    print(os.path.isfile(path + target_name))
                    break
                input_img = normalize(tf.cast(np.load(path + input_name),tf.float32))
                target_img = normalize(tf.cast(np.load(path + target_name),tf.float32))
                this_map_input.append(input_img)
                this_map_target.append(target_img)
                index += 1
            if(len(this_map_input) <= limit):
                for x in this_map_input: inputs.append(x)
                for x in this_map_target: targets.append(x)
            else:
                indexes = random.sample(range(len(this_map_input)), limit)
                for x in indexes: inputs.append(this_map_input[x])
                for x in indexes: targets.append(this_map_target[x])
        print("created dataset of length " + str(len(inputs)))
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
                plt.imshow(display_list[i] * 0.5 + 0.5)
                plt.axis('off')
            plt.show()

    def split_into_train_test(dataset:np.ndarray, ratio=0.9):
        index = int(len(dataset) * ratio)
        return dataset[:index], dataset[index:]

    #inputs, targets = extract_arrays(name_filter=["Fort_Joy_Exterior"], limit=100)

    def create_dataset(inputs, targets):
        return tf.data.Dataset.from_tensor_slices(inputs, targets)

    def save_dataset(dataset, path):
        dataset.save(path)
    #test_images(inputs,targets)

    def rotate_images(inputs: np.ndarray, targets: np.ndarray):
        rotated_targets = []
        rotated_inputs = []
        for i in range(min([len(inputs), len(targets)])):
            for j in range(4):
                inp = inputs[i]
                tar = targets[i]
                rotated_inputs.append(tf.image.rot90(inp, j))
                rotated_targets.append(tf.image.rot90(tar, j))
        return rotated_inputs,rotated_targets

    def discard_alpha(targets : np.ndarray):
        targets_no_alpha = [ x[:,:,:3] for x in targets  ]
        return targets_no_alpha
        
    inputs, targets = extract_arrays(name_filter= None,limit=9999)
    targets = discard_alpha(targets)
    #inputs, targets = rotate_images(inputs,targets)
    inputs = np.asarray(inputs)
    targets = np.asarray(targets)
    print("rotated dataset: " + str(len(inputs)))
    print(inputs.shape)
    print(targets.shape)
    test_images(inputs,targets)
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.1, random_state=42)

    
    with open("./" + DATASET_NAME +"_inputs_train", "wb") as f:
        pickle.dump(inputs_train, f)

    with open("./" + DATASET_NAME +"_inputs_test", "wb") as f:
        pickle.dump(inputs_test, f)
    
    with open("./" + DATASET_NAME +"_targets_train", "wb") as f:
        pickle.dump(targets_train, f)

    with open("./" + DATASET_NAME +"_targets_test", "wb") as f:
        pickle.dump(targets_test, f)
