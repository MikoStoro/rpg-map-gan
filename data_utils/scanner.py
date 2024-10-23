from typing import Any
import PIL
import cv2
from PIL import Image, ImageDraw, ImageFilter
from itertools import product
import numpy as np
from numpy import floating, ndarray, dtype
import tensorflow as tf
import utils

from tensorflow.python.types.data import DatasetV2

GRID_SIZE = 10  # size will probably depend on specific map
RESIZE = 330
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

def _get_rgb_distance(a: tuple, b: tuple) -> floating[Any]:
    point1 = np.array(a)
    point2 = np.array(b)
    return np.linalg.norm(point1 - point2)

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

def _sliding_window_matrices(matrix, window_size=16, step_size=14, replace_value=0)  -> np.ndarray[Any, np.dtype[Any]]:
    padded_matrix = np.pad(matrix, pad_width=window_size // 2, mode='constant', constant_values=replace_value)
    submatrices = []

    range_indices = range(0, matrix.shape[0], step_size)

    for i in range_indices:
        for j in range_indices:
            submatrix = padded_matrix[i:i + window_size, j:j + window_size]
            submatrices.append(submatrix)

    return np.array(submatrices)

def _numpy_arr_to_dataset(arr: np.array) -> DatasetV2:
    return tf.data.Dataset.from_tensor_slices(arr)

def _convert_image_to_cv_matrix(image: Image) -> cv2.typing.MatLike:
    image_array: np.ndarray = np.array(image)
    return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

def scan_map(file_path: str, defined_colors=None, grid_size = None) -> np.ndarray[Any, np.dtype[Any]]:
    if defined_colors is None:
        defined_colors = DEFINED_COLORS
    if grid_size is None:
        grid_size = GRID_SIZE
    with _process_image(file_path) as img:
        w, h = img.size
        tiles = _image_into_grid(img, grid_size)
    rgb_list = []
    for tile in tiles:
        most_common_color = utils.get_most_common_color(tile)
        rgb_list.append(_get_closest_defined_color_symbol(most_common_color, defined_colors))

    result = np.array(rgb_list, dtype=str).reshape(-1, w // grid_size)
    _remove_isolated_pixels(result)
    return result

def get_map_with_scan_overlay(file_path: str, color_matrix: ndarray[Any, dtype[Any]], defined_colors=None, opacity=0.8, grid_size = None) -> cv2.typing.MatLike:
    if defined_colors is None:
        defined_colors = DEFINED_COLORS
    if grid_size is None:
        grid_size = GRID_SIZE
    with _process_image(file_path).convert("RGBA") as img:
        opacity = int(255 * opacity)
        reversed_defined_colors_dict = {value: (key + (opacity,)) for key, value in defined_colors.items()}
        width, height = img.size
        overlay = Image.new("RGBA", (width, height))
        draw = ImageDraw.Draw(overlay)
        matrix_height, matrix_width = color_matrix.shape
        for y in range(matrix_height):
            for x in range(matrix_width):
                fill = reversed_defined_colors_dict.get(color_matrix[y][x])
                draw.rectangle((x * grid_size, y * grid_size, (x + 1) * grid_size, (y + 1) * grid_size), fill=fill)
        image = Image.alpha_composite(img, overlay)
        return _convert_image_to_cv_matrix(image)

def save_map_with_scan_overlay(file_path: str, defined_colors=None) -> None:
    color_matrix = scan_map(file_path, defined_colors)
    result = Image.fromarray(get_map_with_scan_overlay(file_path, color_matrix, defined_colors))
    result.save(file_path + "debug.png")

def serialize_map_submatrices(file_path: str, defined_colors=None, window_size=16, step_size=14, replace_value=0) -> None:
    submatrices =  _sliding_window_matrices(scan_map(file_path, defined_colors), window_size, step_size, replace_value)
    np.save(file_path, submatrices)

def get_scan_map_dataset(file_path: str, defined_colors=None) -> DatasetV2:
    color_matrix = scan_map(file_path, defined_colors)
    dataset = _numpy_arr_to_dataset(color_matrix)
    return dataset


