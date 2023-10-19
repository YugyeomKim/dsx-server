import torch
import sys
import os
import json
from PIL import Image
from torchvision import transforms

dirname = os.path.dirname(__file__)
model = torch.jit.load(os.path.join(dirname, "alexnet_v1.pt"))
model.eval()

imagePathList = json.loads(sys.argv[1])
images = []
for imagePath in imagePathList:
    image = Image.open(imagePath)
    images.append(image)


with torch.no_grad():
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((227, 227), antialias=False),
    ])

    batch = torch.stack([img_transforms(image) for image in images])

    pred = model(batch)

    result = pred.argmax(dim=1).tolist()

print(result)
