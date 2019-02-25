import io
import requests
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch import nn
import matplotlib.pyplot as plt
import math

img_url = 'https://s3.amazonaws.com/outcome-blog/wp-content/uploads/2017/02/25192225/cat.jpg'

def load_and_preprocess_img():
    response = requests.get(img_url)
    img_pil = Image.open(io.BytesIO(response.content))

    normalize = transforms.Normalize(
        mean = [0.4914, 0.4822, 0.4465],
        std = [0.2023, 0.1994, 0.2010]
        )
    preprocess = transforms.Compose([
        transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize
        ])

    img_tensor = preprocess(img_pil)
    img_tensor.unsqueeze_(0)
    #return Variable(img_tensor)
    return img_tensor

def plot_filters(img, activation, num_filters):
    plt.figure(1, figsize=(10,10))
    n_columns = 6
    n_rows = math.ceil(num_filters / n_columns) + 1
    plt.subplot(n_rows, n_columns, 1)
    plt.imshow(img[0,2,:,:])
    for i in range(num_filters):
        plt.subplot(n_rows, n_columns, i+2)
        #plt.title('Filter ' + str(i))
        plt.imshow(activation[0,i,:,:], interpolation="nearest", cmap="gray")


def visualize_first_conv_layer_filters_resnet18(image, num_filters):
    activation = models.resnet18(pretrained=True).conv1(image)
    plot_filters(image, activation.detach().numpy(), num_filters)



def visualize_last_conv_layer_filters_resnet18(image, num_filters):
    resnet18_last_conv = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])

    for param in resnet18_last_conv.parameters():
        param.requires_grad = False

    output = resnet18_last_conv(image)

    plot_filters(output.detach().numpy(), num_filters)

def visualize_weights_first_conv_layer():
    weights = models.resnet18(pretrained=True).conv1.weight.data.numpy()
    plt.imshow(weights[0, 0, :, :])
    
print('Start preprocessing..')
img = load_and_preprocess_img()

#first_layer = visualize_first_conv_layer_filters_resnet18(img)
#plt.imshow(first_layer[0, :, :, 2], cmap='gray')
#visualize_first_conv_layer_filters_resnet18(img, 10)
#plt.show()

#visualize_last_conv_layer_filters_resnet18(img, 10)
#plt.show()

visualize_weights_first_conv_layer()
plt.show()
#Spørsmål - hvordan printe original bilde?
# Noe er galt med weights
