import torch
import torchvision
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 2

# Load model WITHOUT aux_classifier
model = torchvision.models.segmentation.deeplabv3_resnet50(weights=None, aux_loss=False)

# Replace final classifier head with 2-class output
model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

# Load weights
state_dict = torch.load("unet.pth", map_location=device)

# Filter out aux_classifier weights
filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("aux_classifier")}

model.load_state_dict(filtered_state_dict, strict=False)

model.to(device).eval()

# Export ONNX
dummy_input = torch.randn(1, 3, 512, 512).to(device)
torch.onnx.export(
    model,
    dummy_input,
    "unet.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)

print("ONNX model exported successfully!")
