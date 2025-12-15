import json
import numpy as np
import os
from PIL import Image, ImageDraw
from tqdm import tqdm

# === Paths ===
IMAGE_DIR = "images"
ANNOTATION_FILE = "coco/train.json"
MASK_DIR = "masks"
os.makedirs(MASK_DIR, exist_ok=True)

# Load COCO data
with open(ANNOTATION_FILE, 'r') as f:
    coco = json.load(f)

image_info = {img['id']: img for img in coco['images']}

CLASS_MAP = {
    "road": 1,
    "car": 2,
    "pothole": 3,
    "bike": 4,
}

for ann in tqdm(coco['annotations'], desc="Generating masks"):
    img_id = ann['image_id']
    category_id = ann['category_id']

    # get category name from annotation
    class_name = next(cat['name'] for cat in coco['categories'] if cat['id'] == category_id)
    class_value = CLASS_MAP.get(class_name, 0)

    img_meta = image_info[img_id]
    filename = img_meta['file_name']
    width, height = img_meta['width'], img_meta['height']

    mask_path = os.path.join(MASK_DIR, filename.replace(".jpg", ".png").replace(".jpeg", ".png"))

    if os.path.exists(mask_path):
        mask = Image.open(mask_path)
    else:
        mask = Image.new("L", (width, height), 0)

    draw = ImageDraw.Draw(mask)

    seg = ann.get('segmentation', [])
    if isinstance(seg, list) and len(seg) > 0:
        for polygon in seg:
            points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
            draw.polygon(points, outline=class_value, fill=class_value)

    mask.save(mask_path)

print("🎯 Masks generated and saved in:", MASK_DIR)
