import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt

model_path = "weatherAndroid.onnx"
session = ort.InferenceSession(model_path)

input_name = session.get_inputs()[0].name
print("Input Name:", input_name)



image_path = "liveSamp2.jpg" #image 

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (512, 512))

img_input = img_resized.astype(np.float32) / 255.0
img_input = np.transpose(img_input, (2, 0, 1))
img_input = np.expand_dims(img_input, axis=0)

print("Input Shape:", img_input.shape)


#prepare for model
output = session.run(None, {input_name: img_input})[0]
mask = output.squeeze(0).squeeze(0).astype(np.uint8)

print("Mask Shape:", mask.shape)
print("Unique Classes:", np.unique(mask))


colors = np.array([
    [0, 0, 0],       # class 0 - background
    [255, 0, 0],     # class 1 - red    road
    [0, 255, 0],     # class 2 - green
    [0, 0, 255],     # class 3 - blue
    [255, 255, 0]    # class 4 - yellow
], dtype=np.uint8)

color_mask = colors[mask]

color_mask_resized = cv2.resize(color_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)



overlay = (0.6 * img + 0.4 * color_mask_resized).astype(np.uint8) #make overlay

#show on matplot
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Segmentation Mask")
plt.imshow(color_mask_resized)
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Overlay")
plt.imshow(overlay)
plt.axis("off")

plt.tight_layout()
plt.show()



