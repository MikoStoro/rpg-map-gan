import numpy as np
kelly_colors = dict(vivid_yellow=(255, 179, 0),
                    strong_purple=(128, 62, 117),
                    vivid_orange=(255, 104, 0),
                    very_light_blue=(166, 189, 215),
                    vivid_red=(193, 0, 32),
                    grayish_yellow=(206, 162, 98),
                    medium_gray=(129, 112, 102),

                    # these aren't good for people with defective color vision:
                    vivid_green=(0, 125, 52),
                    strong_purplish_pink=(246, 118, 142),
                    strong_blue=(0, 83, 138),
                    strong_yellowish_pink=(255, 122, 92),
                    strong_violet=(83, 55, 122),
                    vivid_orange_yellow=(255, 142, 0),
                    strong_purplish_red=(179, 40, 81),
                    vivid_greenish_yellow=(244, 200, 0),
                    strong_reddish_brown=(127, 24, 13),
                    vivid_yellowish_green=(147, 170, 0),
                    deep_yellowish_brown=(89, 51, 21),
                    vivid_reddish_orange=(241, 58, 19),
                    dark_olive_green=(35, 44, 22))
#labels_path = "./labels.txt"
labels_path = "./data_utils/labels.txt"
def get_single_color_dict(index):
    #ret = {}
    ret = []
    for key,value in kelly_colors.items():
        #ret[key] = value[index]
        ret.append(value[index])
    return ret

kelly_colors_r = get_single_color_dict(0)
kelly_colors_g = get_single_color_dict(1)
kelly_colors_b = get_single_color_dict(2)

def vec_translate(a, my_dict):    
   return np.vectorize(my_dict.__getitem__)(a)

def get_colormap(label_matrix: np.ndarray):
    with open(labels_path, "r") as f:
        labels = f.readlines()
        labels = [ x.strip() for x in labels ]
        print(labels)
    
    label_color_r = {}
    label_color_g = {}
    label_color_b = {}
    for i in range(len(labels)):
        label = labels[i]
        label_color_r[label] = kelly_colors_r[i]
        label_color_g[label] = kelly_colors_g[i]
        label_color_b[label] = kelly_colors_b[i]
       
    colormap_r = vec_translate(label_matrix,label_color_r)
    print("R" + str(colormap_r))
    colormap_g = vec_translate(label_matrix,label_color_g)
    colormap_b = vec_translate(label_matrix,label_color_b)
    return np.dstack((colormap_r,colormap_g,colormap_b)).astype(np.uint8)  


'''x = np.array([3,3])
x = np.asarray([
    ['wood','lava','water','wood','lava','water'],
    ['stone','stone_manmade','none','wood','lava','water'],
    ['wood','lava','water','wood','lava','water']
])

from matplotlib import pyplot as plt
plt.imshow(get_colormap(x), interpolation='nearest')
plt.show()'''

