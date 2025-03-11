import numpy as np
from matplotlib import pyplot as plt

orig = np.load("./Black_Ring_Encampment_Map_ORIGINAL_1092.npy")
colormap = np.load("./Black_Ring_Encampment_Map_COLORMAP_1092.npy")

plt.imshow(orig, interpolation='nearest')
plt.show()
plt.imshow(colormap, interpolation='nearest')
plt.show()