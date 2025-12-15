from PIL import Image
import numpy as np

mask_path = "test/masksTest/AS_9810.png"

img = Image.open(mask_path)
print("Mode:", img.mode)

