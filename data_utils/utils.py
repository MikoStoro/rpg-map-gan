def get_most_common_color(img) -> tuple:
    pixels = list(img.getdata())
    count_dict = {i: pixels.count(i) for i in pixels}
    return max(count_dict, key=count_dict.get)

def get_avg_color(channel, pixels) -> int:
    lst = [colors[channel] for colors in pixels]
    return round(sum(lst) / len(lst))

def get_avg_rgb(img) -> list[int]:
    pixels = list(img.getdata())
    avg_rgb = []
    for channel in range(3):
        avg_rgb.append(get_avg_color(channel, pixels))
    return avg_rgb

def int_to_rgb(value: int) -> tuple:
    if not (0 <= value <= 0xFFFFFF):
        raise ValueError("Value should be between 0 and 0xFFFFFF")
    red = (value >> 16) & 0xFF
    green = (value >> 8) & 0xFF
    blue = value & 0xFF
    return red, green, blue

def translate_colors_dict_to_json(dictionary: dict):
    new_dict = {}
    for key,value in dictionary.items():
        print("dict to json: dict key:  " + str(key) )
        key = list(key)
        print("new key:" + str(key))
        newKey = str(key[0]) + " " + str(key[1]) + " " + str(key[2])
        print("dict to json: json key:  " + str(newKey) )
        new_dict[newKey] = value
    return new_dict

def translate_json_to_colors_dict(json_dict : dict):
    new_dict = {}
    for key,value in json_dict.items():
        print("json to dict: json key:" + key)
        newKey = str(key).split(" ")

        newKey = tuple([int(x) for x in newKey])
        print(newKey)
        print("json to dict: dict key:" + str(newKey))
        new_dict[newKey] = value
    return new_dict