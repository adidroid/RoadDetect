import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import matplotlib.pyplot as plt


COLORS = {
    0: (0, 255, 0),      # road = green
    1: (255, 0, 0),      # car = blue
    2: (0, 0, 255),      # pothole = red
    3: (255, 255, 0),    # bike = yellow
    4: (0, 0, 0)         # background = black
}

def preprocess_frame(frame):
    img = cv2.resize(frame, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    return img


def mask_to_color(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in COLORS.items():
        color_mask[mask == cls] = color
    return color_mask


def overlay_mask(frame, mask, alpha=0.5):
    mask_color = mask_to_color(mask)
    mask_color = cv2.resize(mask_color, (frame.shape[1], frame.shape[0]))
    overlay = cv2.addWeighted(frame, 1 - alpha, mask_color, alpha, 0)
    return overlay


session = ort.InferenceSession("weatherAdjust.onnx")


video_path = "rooo.mp4"
cap = cv2.VideoCapture(video_path)

plt.ion()  
fig, ax = plt.subplots()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    inp = preprocess_frame(frame)

    out = session.run(None, {"input": inp})
    logits = out[0]
    pred = np.argmax(logits, axis=1)[0]

    overlay = overlay_mask(frame, pred)

    # Display with matplotlib
    ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax.set_title("Segmentation Output")
    ax.axis('off')
    plt.pause(0.001)
    ax.clear()

cap.release()
plt.ioff()
plt.show()
