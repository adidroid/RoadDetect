import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


COLOR_MAP = {
    (0, 0, 255): 0,     # Blue = Road
    (255, 255, 0): 1,   # Yellow = Vehicle
}


def mask_to_class(mask):
    mask = np.array(mask)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
    for rgb, idx in COLOR_MAP.items():
        label_mask[(mask == rgb).all(axis=2)] = idx
    return label_mask


class RoadVehicleDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))
        self.transform = transform

    def __len__(self):  
        return len(self.images)

    def __getitem__(self, idx):  
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        mask = mask.resize((512, 512), resample=Image.NEAREST)
        mask = torch.from_numpy(mask_to_class(mask)).long()

        return image, mask

        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        mask = mask.resize((512, 512), resample=Image.NEAREST)
        mask = torch.from_numpy(mask_to_class(mask)).long()

        return image, mask
    




transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = RoadVehicleDataset("images", "masks", transform=transform)
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)



num_classes = 2

model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

model = model.to(device)



criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=1e-4)


ds = RoadVehicleDataset("images", "masks", transform=transform)
img, mask = ds[0]
print(img.shape, mask.shape)


epochs = 15

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        outputs = model(imgs)['out']
        
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.4f}")




torch.save(model.state_dict(), "road_vehicle_model.pth")

dummy_input = torch.randn(1, 3, 512, 512).to(device)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "road_vehicle_segmentation.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    dynamic_axes={
        "input": {0: "batch"},
        "output": {0: "batch"}
    }
)

print("ONNX model exported successfully!")


