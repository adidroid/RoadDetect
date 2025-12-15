import os
import numpy as np
from PIL import Image

img_dir = "train/imagesTrain"
mask_dir = "train/masksTrain"

for f in sorted(os.listdir(img_dir)):
    img_path = os.path.join(img_dir, f)

    ext = os.path.splitext(f)[1]
    mask_name = f.replace(ext, ".png")
    mask_path = os.path.join(mask_dir, mask_name)

    if not os.path.exists(mask_path):
        print("❌ Mask missing:", mask_path)
        continue

    img = np.array(Image.open(img_path))
    mask = np.array(Image.open(mask_path))

    if img.shape[:2] != mask.shape[:2]:
        print("❌ Size mismatch:", f)
        print("   Image:", img.shape)
        print("   Mask:", mask.shape)
