import os
from typing import Any

import PIL
from PIL import Image, ImageDraw, ImageFilter
from itertools import product
import numpy as np
from numpy import floating, ndarray, dtype
import tensorflow as tf
import cv2
import json

GRID_SIZE = 8  # size will probably depend on specific map
RESIZE = 480
MAP_PATH = "./maps"
OUTPUT_PATH = "./output"

DEFINED_COLORS = {
    (128, 128, 128): 'S',  # Gray for stone
    (0, 0, 0): 'K',  # Black for black
    (50, 50, 255): 'W',  # Blue for water
    (139, 69, 19): 'D',  # Brown for wood
    (50, 140, 50): 'G'  # Green for grass
}


def _process_image(file_path):
    return Image.open(file_path).resize((RESIZE, RESIZE)).filter(ImageFilter.BLUR)


def _get_closest_defined_color_symbol(color: tuple, defined_colors: dict) -> str:
    distances = {_get_rgb_distance(color, k): k for k in defined_colors.keys()}
    return defined_colors.get(distances.get(min(distances.keys())))


def _int_to_rgb(value: int) -> tuple:
    if not (0 <= value <= 0xFFFFFF):
        raise ValueError("Value should be between 0 and 0xFFFFFF")

    red = (value >> 16) & 0xFF
    green = (value >> 8) & 0xFF
    blue = value & 0xFF

    return red, green, blue


def _get_rgb_distance(a: tuple, b: tuple) -> floating[Any]:
    point1 = np.array(a)
    point2 = np.array(b)
    return np.linalg.norm(point1 - point2)


def _get_avg_color(channel, pixels):
    lst = [colors[channel] for colors in pixels]
    return round(sum(lst) / len(lst))


def _get_avg_rgb(img):
    pixels = list(img.getdata())
    avg_rgb = []
    for channel in range(3):
        avg_rgb.append(_get_avg_color(channel, pixels))
    return avg_rgb


def _get_most_common_color(img):
    pixels = list(img.getdata())
    count_dict = {i: pixels.count(i) for i in pixels}
    return max(count_dict, key=count_dict.get)


def _image_into_grid(image: PIL.Image.Image, d: int):
    w, h = image.size
    grid = product(range(0, h - h % d, d), range(0, w - w % d, d))
    output = []
    for i, j in grid:
        box = (j, i, j + d, i + d)
        cropped_img = image.crop(box)
        output.append(cropped_img)
    return output


def _remove_isolated_pixels(color_matrix: ndarray[Any, dtype[Any]]) -> None:
    rows, cols = color_matrix.shape
    for i in range(rows):
        for j in range(cols):
            neighbor = ""

            if i > 0:
                if neighbor == "":
                    neighbor = color_matrix[i - 1, j]
                if neighbor != color_matrix[i - 1, j]:
                    continue

                # Check below
            if i < rows - 1:
                if neighbor == "":
                    neighbor = color_matrix[i + 1, j]
                if neighbor != color_matrix[i + 1, j]:
                    continue

                # Check left
            if j > 0:
                if neighbor == "":
                    neighbor = color_matrix[i, j - 1]
                if neighbor != color_matrix[i, j - 1]:
                    continue

                # Check right
            if j < cols - 1:
                if neighbor == "":
                    neighbor = color_matrix[i, j + 1]
                if neighbor != color_matrix[i, j + 1]:
                    continue
            color_matrix[i, j] = neighbor

def _create_debug_map(file_path: str, color_matrix: ndarray[Any, dtype[Any]], defined_colors=None, opacity=0.8):
    if defined_colors is None:
        defined_colors = DEFINED_COLORS
    with _process_image(file_path).convert("RGBA") as img:
        width, height = img.size
        opacity = int(255 * opacity)
        reversed_defined_colors_dict = {value: (key + (opacity,)) for key, value in defined_colors.items()}
        overlay = Image.new("RGBA", (width, height))
        draw = ImageDraw.Draw(overlay)
        matrix_height, matrix_width = color_matrix.shape
        for y in range(matrix_height):
            for x in range(matrix_width):
                fill = reversed_defined_colors_dict.get(color_matrix[y][x])
                draw.rectangle((x * GRID_SIZE, y * GRID_SIZE, (x + 1) * GRID_SIZE, (y + 1) * GRID_SIZE), fill=fill)

        output = Image.alpha_composite(img, overlay)
    output.save(file_path + "debug.png")

def _sliding_window_matrices(matrix, window_size=16, step_size=14, replace_value=0):
    padded_matrix = np.pad(matrix, pad_width=window_size // 2, mode='constant', constant_values=replace_value)
    submatrices = []

    range_indices = range(0, matrix.shape[0], step_size)

    for i in range_indices:
        for j in range_indices:
            submatrix = padded_matrix[i:i + window_size, j:j + window_size]
            submatrices.append(submatrix)

    return np.array(submatrices)


def _create_color_matrix(file_path: str, defined_colors=None) -> np.ndarray[Any, np.dtype[Any]]:
    if defined_colors is None:
        defined_colors = DEFINED_COLORS
    with _process_image(file_path) as img:
        w, h = img.size
        tiles = _image_into_grid(img, GRID_SIZE)
    rgb_list = []
    for tile in tiles:
        most_common_color = _get_most_common_color(tile)
        rgb_list.append(_get_closest_defined_color_symbol(most_common_color, defined_colors))

    return np.array(rgb_list, dtype=str).reshape(-1, w // GRID_SIZE)


def _numpy_arr_to_dataset(arr: np.array):
    return tf.data.Dataset.from_tensor_slices(arr)

#

def debug_scan_map(file_path: str, defined_colors=None):
    color_matrix = scan_map(file_path, defined_colors)
    _create_debug_map(file_path, color_matrix, defined_colors)

def serialize_map_submatrices(file_path: str, defined_colors=None, window_size=16, step_size=14, replace_value=0):
    submatrices =  _sliding_window_matrices(_create_color_matrix(file_path, defined_colors), window_size, step_size, replace_value)
    np.save(file_path, submatrices)

def scan_map(file_path: str, defined_colors=None):
    color_matrix = _create_color_matrix(file_path, defined_colors)
    _remove_isolated_pixels(color_matrix)

    dataset = _numpy_arr_to_dataset(color_matrix)
    return dataset


#









def create_mouse_event():
    coords = []

    def mouse_press_get_coords_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            coords.append((x, y))

    return mouse_press_get_coords_event, coords


def create_json_with_colors_and_items(path):
    img = cv2.imread(path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('image', img)

    mouse_press_get_coords_event, coords = create_mouse_event()
    cv2.setMouseCallback('image', mouse_press_get_coords_event)
    cv2.waitKey()

    colors_dict = {}
    for i in range(0, len(coords), 2):
        name = input("Enter a name: ")
        cropped_img = img[coords[i][1]:coords[i + 1][1], coords[i][0]:coords[i + 1][0]]
        print(str(coords[i][0]) + '\n')

        print(coords[i + 1][0])
        cropped_img = Image.fromarray(cropped_img)
        cropped_img.show()
        rgb = _get_most_common_color(cropped_img)
        colors_dict[name] = rgb

    with open(OUTPUT_PATH + '/coords.json', 'w') as f:
        json.dump(colors_dict, f)
    cv2.destroyAllWindows()
