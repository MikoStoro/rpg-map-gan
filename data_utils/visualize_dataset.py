import matplotlib.pyplot as plt
import sys
import numpy as np
import random
import dataSetCreator as dsc

print(sys.argv)


#filename = sys.argv[1]
#dim = int(sys.argv[2])

inputs, targets = dsc.extract_arrays(limit = 1000, path="./debug_results/")

for i in range(50):
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