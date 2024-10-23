from PIL import Image
import utils
import cv2
import json

GRID_SIZE = 5
MAP_PATH = "./maps"
OUTPUT_PATH = "./output"
DEFINED_COLORS = {(255, 0, 0): 4, (0, 255, 0): 3, (0, 0, 255): 2, (255, 255, 255): 1, (0, 0, 0): 0}


def create_mouse_event():
    coords = []

    def mouse_press_get_coords_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            coords.append((x, y))
            print(f"selected: ({x}, {y})")

    return mouse_press_get_coords_event, coords


def get_slice(path, x1,y1,x2,y2):
    img = cv2.imread(path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cropped_img = img[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
    cropped_img = Image.fromarray(cropped_img)
    cropped_img.show()

def create_json_with_colors_and_items(path):
    img = cv2.imread(path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('image', img)
    mouse_press_get_coords_event, coords = create_mouse_event()
    cv2.setMouseCallback('image', mouse_press_get_coords_event)

    colors_dict = {}
    while True:
        print("Click two points to select a region, confirm with any button and type a name, or press 'q' to quit.")

        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()
            return

        x1, y1 = coords[0]
        x2, y2 = coords[1]
        cropped_img = img[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]

        cropped_img = Image.fromarray(cropped_img)
        cropped_img.show()

        name = input("Enter a name:")
        rgb = utils.get_most_common_color(cropped_img)
        colors_dict[name] = rgb
        print(f"Added {name}")

        with open('coords.json', 'w') as f:     #Overwrites everytime, so work won't be lost in case of an error
            json.dump(colors_dict, f)

        coords.clear()


#create_json_with_colors_and_items('../maps/Fort_Joy_Ground_Floor-1.png')
