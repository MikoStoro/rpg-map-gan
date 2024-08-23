from data_utils.scanner import Image,_get_closest_defined_color_symbol,_get_most_common_color,_create_color_matrix,np,_remove_isolated_pixels,_sliding_window_matrices,debug_scan_map,serialize_map_submatrices

def test_get_most_common_color():
    img = Image.open("../maps/test/test_common_color.jpg")
    a = _get_most_common_color(img)
    assert a == (4, 50, 100)


def test_get_closest_defined_color_symbol():
    defined_colors = {(255, 0, 0): 4, (0, 255, 0): 3, (0, 0, 255): 2}
    color0 = (255, 254, 254)
    color1 = (0, 255, 0)
    color2 = (0, 254, 255)
    assert _get_closest_defined_color_symbol(color0, defined_colors) == 4
    assert _get_closest_defined_color_symbol(color1, defined_colors) == 3
    assert _get_closest_defined_color_symbol(color2, defined_colors) == 2


def test_create_color_matrix():
    defined_colors = {(255, 0, 1): 4, (1, 255, 0): 3, (1, 0, 255): 2}
    file = "../maps/test/colored_grid.png"
    output = _create_color_matrix(file, defined_colors)

    assert output[0][0] == '4'
    assert output[0][45] == '3'
    assert output[20][0] == '3'
    assert output[40][0] == '2'
    assert output[59][59] == '4'


def test_remove_isolated_pixels():
    array = np.array([['B', 'A', 'B', 'C'],
                      ['A', 'C', 'A', 'A'],
                      ['A', 'A', 'A', 'B'],
                      ['B', 'A', 'A', 'A']])

    expected = np.array([['A', 'A', 'B', 'C'],
                         ['A', 'A', 'A', 'A'],
                         ['A', 'A', 'A', 'A'],
                         ['A', 'A', 'A', 'A']])

    _remove_isolated_pixels(array)

    assert np.array_equal(expected, array)


def test_sliding_window_submatrices():
    matrix = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])
    window_size = 3
    step_size = 2
    replace_value = 0

    expected_result = np.array([
        [
            [0, 0, 0],
            [0, 1, 2],
            [0, 5, 6]
        ],
        [
            [0, 0, 0],
            [2, 3, 4],
            [6, 7, 8]
        ],
        [
            [0, 5, 6],
            [0, 9, 10],
            [0, 13, 14]
        ],
        [
            [6, 7, 8],
            [10, 11, 12],
            [14, 15, 16]
        ]
    ])

    result = _sliding_window_matrices(matrix, window_size, step_size, replace_value)
    assert np.array_equal(result, expected_result)

# def test_mouse_select():
#     create_json_with_colors_and_items('maps/big_colored_grid.png')
#     f = open('coords.json')
#     data = json.load((f))
#     assert data["red"] == [255,0,0]

def test_serialize_map_submatrices():
    original = _sliding_window_matrices(file_path="../maps/test/colored_grid.png")
    serialize_map_submatrices(file_path="../maps/test/colored_grid.png")
    loaded = np.load("../maps/test/colored_grid.png.npy")
    assert np.equal(original,loaded)

def test_debug_scan_map():
    debug_scan_map("../maps/Fort_Joy_Harbour_map.png")
    debug_scan_map("../maps/Underground_Tunnel_Map.png")