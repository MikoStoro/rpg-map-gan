import matplotlib.pyplot as plt
import sys
import numpy as np
import random

print(sys.argv)


filename = sys.argv[1]
dim = int(sys.argv[2])

dataset = np.load(filename)


plt.figure(figsize=(10.0, 10.0))
index = 0
for i in range(0,16):
    # Get image and reshape
    image = dataset[random.randint(0, 1000)]
    image = np.reshape(image, (dim,dim))
    image = image.astype('float32')
    # Plot
    plt.subplot(4, 4, index+1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    index += 1
plt.show()