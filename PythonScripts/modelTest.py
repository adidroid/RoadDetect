import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


COLORS = {
    0: (0, 255, 0),      # background = green
    1: (255, 0, 0),      # car = blue
    2: (0, 0, 255),      # actually road
    3: (255, 255, 0),    # bike = cyan
    4: (0, 0, 0)         # not sure = black
}




def preprocess(path):
    img = Image.open(path).convert("RGB").resize((512, 512))
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = img_np.transpose(2, 0, 1)
    img_np = np.expand_dims(img_np, axis=0)
    return img_np, np.array(img)




def mask_to_color(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for cls, color in COLORS.items():
        color_mask[mask == cls] = color

    return color_mask




def overlay_mask_on_image(orig_img, mask, alpha=0.5):
    orig_img = cv2.resize(orig_img, (mask.shape[1], mask.shape[0]))
    color_mask = mask_to_color(mask)
    overlay = cv2.addWeighted(orig_img, 1 - alpha, color_mask, alpha, 0)
    return overlay




session = ort.InferenceSession("fastscnn_5class.onnx")

img_input, original = preprocess("train/newTrimg/AX_7050.jpeg")    
outputs = session.run(None, {"input": img_input})

logits = outputs[0]               # shape: [1, 5, 512, 512]
pred = np.argmax(logits, axis=1)[0]   # (512, 512)



#show mask from this code here :D
overlay = overlay_mask_on_image(original, pred)

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(original)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Segmentation Mask")
plt.imshow(mask_to_color(pred))
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(overlay)
plt.axis("off")

plt.tight_layout()
plt.show()


