import torch
import sys
import os
from PIL import Image
from torchvision import transforms

dirname = os.path.dirname(__file__)
model = torch.jit.load(os.path.join(dirname, "alexnet_v1.pt"))

image = Image.open(sys.argv[1])

img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((227, 227), antialias=False),
])

img_input = img_transforms(image)
img_input = img_input.unsqueeze(0)

pred = model(img_input)
print(pred.argmax().item())
