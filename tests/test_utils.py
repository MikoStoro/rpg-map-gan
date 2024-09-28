from data_utils.utils import get_most_common_color
from PIL import Image



def test_get_most_common_color():
    img = Image.open("../maps/test/test_common_color.jpg")
    a = get_most_common_color(img)
    assert a == (4, 50, 100)