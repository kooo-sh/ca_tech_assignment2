from fastapi import FastAPI, File, UploadFile
from torchvision import models
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import requests
from typing import List
import torch.nn.functional as F
from pydantic import BaseModel

app = FastAPI()

class PredictionResult(BaseModel):
    classification_results: List[str]
    score: List[float]

class PredictionResponse(BaseModel):
    predictions: List[PredictionResult]

@app.post("/predict", response_model=PredictionResponse)
async def check_model(image: UploadFile = File(None), uri: str=None):
    model = models.vgg16(pretrained=True)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, 10)
    model.load_state_dict(torch.load('./vgg16_out10.pth'))

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    if image:
        img = Image.open(io.BytesIO(await image.read()))
    elif uri:
        if uri.startswith("http"):
            response = requests.get(uri)
            img = Image.open(io.BytesIO(response.content))
        else:
            img = Image.open(uri)

    img = transform(img).unsqueeze(0)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)
        score = torch.max(F.softmax(outputs, dim=1)).item()
        predicted = predicted.item()
        label = classes[predicted]
    return PredictionResponse(predictions=[PredictionResult(classification_results=[label], score=[score])])