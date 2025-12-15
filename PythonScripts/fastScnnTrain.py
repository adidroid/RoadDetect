import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch.nn.functional as F



# DATASET
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.files = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_name = img_name.replace(".jpeg", ".png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        #  Load image and mask 
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path), dtype=np.uint8)

        # Apply Albumentations
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]    # float32 tensor (ToTensorV2)
            mask = aug["mask"]      # long tensor

        #  Ensure types 
        image = image.float()       
        mask = mask.long()

        return image, mask



# TRANSFORMS

train_tf = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.4),
    
    # Convert to float32 and tensor
    A.Normalize(mean=(0,0,0), std=(1,1,1)),  
    ToTensorV2()
])

val_tf = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0,0,0), std=(1,1,1)),
    ToTensorV2()
])



# DATA LOADERS

train_ds = SegDataset("train/newTrimg", "train/masksTrain", train_tf)
val_ds   = SegDataset("test/newTesimg",  "test/masksTest",  val_tf)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)


# FASTSCNN MODEL

class FastSCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        # Learning to Downsample
        self.downsample = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 48, 3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),

            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Global Feature Extractor
        self.gfe = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1)
        )

        # Feature Fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(192, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, num_classes, 1)
        )

    def forward(self, x):
        size = x.size()[2:]

        down = self.downsample(x)
        gfe = self.gfe(down)
        gfe_up = F.interpolate(gfe, size=down.size()[2:], mode="bilinear", align_corners=False)

        fused = torch.cat([down, gfe_up], dim=1)
        fused = self.fuse(fused)

        out = self.classifier(fused)
        out = F.interpolate(out, size=size, mode="bilinear", align_corners=False)

        return out

# TRAINING SETUP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FastSCNN(num_classes=5).to(device)

criterion = nn.CrossEntropyLoss()   # mask values must be 0-4
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



# TRAIN LOOP
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, masks)
        loss.backward()
        optimizer.step()

        total += loss.item()

    return total / len(loader)


def eval_one_epoch(model, loader, criterion):
    model.eval()
    total = 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            out = model(imgs)
            loss = criterion(out, masks)
            total += loss.item()

    return total / len(loader)



EPOCHS = 69
best_loss = float("inf")

for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss = eval_one_epoch(model, val_loader, criterion)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train {train_loss:.4f} | Val {val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "fastscnn_best_5class.pth")
        print("✓ Saved best model")


#export model here
model.eval()
dummy = torch.randn(1, 3, 512, 512).to(device)

torch.onnx.export(
    model,
    dummy,
    "fastscnn_5class.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    do_constant_folding=True,
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)

print("Exported to fastscnn_5class.onnx")








