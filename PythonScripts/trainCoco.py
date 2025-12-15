import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png").replace(".jpeg", ".png"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize((512, 512), Image.BILINEAR)
        mask = mask.resize((512, 512), Image.NEAREST)

        mask = np.array(mask).astype(np.int64)

    
        mask[mask == 0] = 1

        if self.transform:
            image = self.transform(image)

        mask = torch.from_numpy(mask).long()
        return image, mask



transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

train_dataset = SegmentationDataset("train/imagesTrain", "train/masksTrain", transform=transform)
test_dataset  = SegmentationDataset("test/testImages", "test/masksTest", transform=transform)


train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False,num_workers=0)

print("Train size:", len(train_dataset))
print("Test size:", len(test_dataset))



class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)

        self.down1 = DoubleConv(3, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

        self.middle = DoubleConv(512, 1024)

        self.up4 = DoubleConv(1024 + 512, 512)
        self.up3 = DoubleConv(512 + 256, 256)
        self.up2 = DoubleConv(256 + 128, 128)
        self.up1 = DoubleConv(128 + 64, 64)

        self.out_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        c1 = self.down1(x)
        c2 = self.down2(self.maxpool(c1))
        c3 = self.down3(self.maxpool(c2))
        c4 = self.down4(self.maxpool(c3))

        m = self.middle(self.maxpool(c4))

        u4 = nn.functional.interpolate(m, scale_factor=2, mode="bilinear", align_corners=True)
        u4 = torch.cat([u4, c4], dim=1)
        u4 = self.up4(u4)

        u3 = nn.functional.interpolate(u4, scale_factor=2, mode="bilinear", align_corners=True)
        u3 = torch.cat([u3, c3], dim=1)
        u3 = self.up3(u3)

        u2 = nn.functional.interpolate(u3, scale_factor=2, mode="bilinear", align_corners=True)
        u2 = torch.cat([u2, c2], dim=1)
        u2 = self.up2(u2)

        u1 = nn.functional.interpolate(u2, scale_factor=2, mode="bilinear", align_corners=True)
        u1 = torch.cat([u1, c1], dim=1)
        u1 = self.up1(u1)

        return self.out_conv(u1)

model = UNet(num_classes=5).to(device)



class_weights = torch.tensor([0.0, 1.0, 1.5, 4.0, 3.0]).to(device)  # 0 unused
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-4)



epochs = 60
best_loss = float("inf")

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)

        outputs = model(imgs)

        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        torch.save(model.state_dict(), "best_unet.pth")
        best_loss = avg_loss

print("🎯 Training Complete. Best Loss:", best_loss)



model.eval()
dummy_input = torch.randn(1, 3, 512, 512).to(device)

torch.onnx.export(
    model,
    dummy_input,
    "unet_road_car_bike_pothole.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)

print("📦 ONNX model exported: unet_road_car_bike_pothole.onnx")

