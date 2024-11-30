import random
import numpy as np
from colormap_createor import get_colormap
file = open("/home/mikostoro/Documents/GitHub/rpg-map-gan/data_utils/labels.txt")

labels = file.readlines()
file.close()
labels = [ x.strip() for x in labels ]
print(labels)

class Point:
    
    x = None
    y = None
    value = 0
    
    def __init__(self,value, x,y):
        self.x = x
        self.y=y 
        self.value = value
    
    def get_children(self):
        children = []
        children.append( Point(self.value,self.x+1, self.y))
        children.append( Point(self.value,self.x-1, self.y))
        children.append( Point(self.value,self.x, self.y+1))
        children.append( Point(self.value,self.x, self.y-1))
        return children
    
    def get_children_and_self(self):
        children = self.get_children()
        children.append(self)
        return children
        

def get_map(x=128, y=128, points=5, classes=5, labels_list = None):
    num_of_points = points
    num_of_calsses = classes
    if labels_list is None:
        labels_list = labels
    x_dim = x
    y_dim = y
    
    
    board = np.full((x_dim,y_dim), "" , dtype=object)
    active_points = []
    for i in range(num_of_points):
        #active_points += Point(np.random.randint(1,num_of_calsses+1),np.random.randint(x_dim), np.random.randint(y_dim)).get_children_and_self()
        active_points += Point(np.random.choice(labels_list),np.random.randint(x_dim), np.random.randint(y_dim)).get_children_and_self()

    while len(active_points) > 0:
        current_index = np.random.randint(len(active_points))
        current_point = active_points[current_index]
        x = current_point.x
        y = current_point.y
        
        if x >= 0 and x < x_dim and \
        y >= 0 and y < y_dim \
        and board[x][y] == "":
            board[x][y] = current_point.value
            active_points += current_point.get_children()
        active_points.pop(current_index)
    return board


def get_maps(n, x=128,y=128,points=5,classes=5):
    return [ get_map(x,y,points,classes) for _ in range(n) ]

def upsample_map(input_map, scale=2):
    return np.kron(input_map, np.ones((scale,scale)))

def upsample_maps(map_array, scale):
    return [  upsample_map(m, scale) for m in map_array  ]

def get_input(labels_list = None):
    return get_colormap(get_map(x=256,y=256,points=5,classes=5, labels_list=labels_list))














