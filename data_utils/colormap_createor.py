import numpy as np
color_dict = dict(  dirt=(255, 179, 0),
                    grass=(128, 62, 117),
                    plant=(255, 104, 0),
                    stone=(166, 189, 215),
                    metal=(193, 0, 32),
                    magic=(206, 162, 98),
                    sand=(129, 112, 102),
                    wood=(0, 125, 52),
                    wood_manmade=(246, 118, 142),
                    stone_manmade=(0, 83, 138),
                    sand_manmade=(255, 122, 92),
                    roof=(83, 55, 122),
                    water=(255, 142, 0),
                    lava=(179, 40, 81),
                    none=(244, 200, 0))

'''color_dict_2 = dict( dirt=(204, 153, 0),
    grass=(0, 255, 0),
    plant=(0,128,0),
    stone=(128, 128, 128),
    metal=(77,77,77),
    magic=(102, 0, 255),
    sand=(255, 255, 0),
    wood=(153, 102, 51),
    wood_manmade=(102, 51, 0),
    stone_manmade=(102, 102, 102),
    sand_manmade=(255, 204, 0),
    roof=(255, 51, 0),
    water=(0,0,255),
    lava=(255, 102, 0),
    none=(0, 0, 0))
'''


labels_path = "./labels.txt"
labels_path2 = "./data_utils/labels.txt"
def get_single_color_dict(index):
    ret = []
    for key,value in color_dict.items():
        ret.append(value[index])
    return ret

color_dict_r = get_single_color_dict(0)
color_dict_g = get_single_color_dict(1)
color_dict_b = get_single_color_dict(2)

def vec_translate(a, my_dict):    
   return np.vectorize(my_dict.__getitem__)(a)

'''def translate_to_new_dict(img: np.ndarray):
    for i in range(len(color_dict)):'''

def get_colormap(label_matrix: np.ndarray):
    try:
        file = open(labels_path)
    except:
        file = open(labels_path2)

    labels = file.readlines()
    file.close()
    labels = [ x.strip() for x in labels ]
    
    label_color_r = {}
    label_color_g = {}
    label_color_b = {}
    for i in range(len(labels)):
        label = labels[i]
        label_color_r[label] = color_dict_r[i]
        label_color_g[label] = color_dict_g[i]
        label_color_b[label] = color_dict_b[i]
       
    colormap_r = vec_translate(label_matrix,label_color_r)
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

