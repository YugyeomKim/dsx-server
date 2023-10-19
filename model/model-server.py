from flask import Flask
from flask import request
import torch
import os
from PIL import Image
from torchvision import transforms

dirname = os.path.dirname(__file__)
model = torch.jit.load(os.path.join(dirname, "alexnet_v1.pt"))
model.eval()

img_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((227, 227), antialias=False),
    ]
)

app = Flask(__name__)


@app.route("/", methods=["POST"])
def hello_world():
    imagePathList = request.get_json()
    images = []
    for imagePath in imagePathList:
        image = Image.open(imagePath)
        images.append(image)

    with torch.no_grad():
        batch = torch.stack([img_transforms(image) for image in images])

        pred = model(batch)

        result = pred.argmax(dim=1).tolist()

    print(result)
    return result


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="127.0.0.1", port=3001)
