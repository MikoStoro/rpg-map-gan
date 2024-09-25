import os
from typing import Any

import numpy
from PIL import Image
from itertools import product
import numpy as np
from numpy import floating
import tensorflow as tf
import cv2
import json
import scanner
GRID_SIZE = 5  # size will probably depend on specific map
MAP_PATH = "./maps"
OUTPUT_PATH = "./output"
DEFINED_COLORS = {(255, 0, 0): 4, (0, 255, 0): 3, (0, 0, 255): 2, (255, 255, 255): 1, (0, 0, 0): 0}



def create_mouse_event():
    coords = []
    def mouse_press_get_coords_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            coords.append((x,y))
    return mouse_press_get_coords_event, coords

def create_json_with_colors_and_items(path):
    img = cv2.imread(path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('image', img)

    mouse_press_get_coords_event, coords = create_mouse_event()
    cv2.setMouseCallback('image', mouse_press_get_coords_event)
    cv2.waitKey()

    colors_dict ={}
    for i in range(0,len(coords),2):
        name = input("Enter a name: ")
        cropped_img = img[coords[i][1]:coords[i+1][1], coords[i][0]:coords[i+1][0]]
        print(str(coords[i][0])+'\n')

        print(coords[i+1][0])
        cropped_img = Image.fromarray(cropped_img)
        cropped_img.show()
        rgb = scanner._get_most_common_color(cropped_img)
        colors_dict[name]=rgb

    with open(OUTPUT_PATH+'/coords.json', 'w') as f:
        json.dump(colors_dict, f)
    cv2.destroyAllWindows()

def create_mouse_event():
    coords = []
    def mouse_press_get_coords_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            coords.append((x, y))
            print(f"selected: ({x}, {y})")
    return mouse_press_get_coords_event, coords

#
# def create_json_with_colors_and_items(path):
#     img = cv2.imread(path, 1)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     cv2.imshow('image', img)
#
#     mouse_press_get_coords_event, coords = create_mouse_event()
#     cv2.setMouseCallback('image', mouse_press_get_coords_event)
#     # while len(coords) % 2 != 0 or len(coords) != 0:
#     #     print("Select two points")
#     #     cv2.waitKey(1)
#
#     colors_dict = {}
#     while True:
#         print("Click two points to select a region, or 'q' to quit.")
#
#         # while len(coords) < 2:
#         key = cv2.waitKey(0)
#         if key == 'q':
#             with open('coords.json', 'w') as f:
#                 json.dump(colors_dict, f)
#             cv2.destroyAllWindows()
#
#         print(f"Selected points: {coords[0]} and {coords[1]}")
#
#         x1, y1 = coords[0]
#         x2, y2 = coords[1]
#         cropped_img = img[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
#
#         cropped_img = Image.fromarray(cropped_img)
#         cropped_img.show()
#
#         name = input("Enter a name:")
#         rgb = utils._get_most_common_color(cropped_img)
#         colors_dict[name] = rgb
#         print(f"Added {name}")
#
#         coords.clear()



create_json_with_colors_and_items('../maps/Fort_Joy_Ground_Floor-1.png')