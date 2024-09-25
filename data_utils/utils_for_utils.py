def _get_most_common_color(img) -> tuple:
    pixels = list(img.getdata())
    count_dict = {i: pixels.count(i) for i in pixels}
    return max(count_dict, key=count_dict.get)
