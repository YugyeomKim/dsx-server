import torch
import sys
import os
import json
import time
from PIL import Image
from torchvision import transforms

# # Measure the elapsed time for loading the model
# start_time = time.time()

dirname = os.path.dirname(__file__)
model = torch.jit.load(os.path.join(dirname, "alexnet_v1.pt"))
model.eval()

# load_model_time = time.time() - start_time
# start_time = time.time()

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

    # transform_time = time.time() - start_time
    # start_time = time.time()

    pred = model(batch)

    result = pred.argmax(dim=1).tolist()

# prediction_time = time.time() - start_time
# with open("elapsed_times.txt", "a") as f:
#     f.write(f"Load model time: {load_model_time:.3f}s\n")
#     f.write(f"Transform data time: {transform_time:.3f}s\n")
#     f.write(f"Predict time: {prediction_time:.3f}s\n")

print(result)
