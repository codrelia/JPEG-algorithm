import numpy
from PIL import Image

from Jpeg import Jpeg

a = Jpeg('4.jpg', 5)
a.save_compressed_image('4_compressed_5')
a.save_image_difference('4_diff_5')
