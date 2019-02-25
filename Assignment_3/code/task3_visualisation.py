import io
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models, transforms
import requests
from PIL import Image

img_url = 'https://s3.amazonaws.com/outcome-blog/wp-content/uploads/2017/02/25192225/cat.jpg'

def load_and_preprocess_img():
    response = request.get(img_url)
    img_pil = Image.open(io.BytesIO(response.content))

    normalize = transforms.Normalize(
        mean = [0.4914, 0.4822, 0.4465],
        std = [0.2023, 0.1994, 0.2010]
        )
    preprocess = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
        ])

    img_tensor = preprocess(img_pil)
    #img_tensor.unsqueeze_(0)
    return img_tensor

def visualize_first_filter_resnet18(image):
    activation = torchvision.models.resnet18(pretrained=True).conv1(image)
    #torchVision save utils
    return activation.view(activation.shape[1], 1, activation.shape[2:])[:10]

def visualize_last_filter_resnet18(image):
    pass

print('Start preprocessing..')
tensor = load_and_preprocess_img()
first_layer = visualize_first_filter_resnet18(tensor)

