import io
import requests
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch import nn
import matplotlib.pyplot as plt
import math

#img_url = 'https://s3.amazonaws.com/outcome-blog/wp-content/uploads/2017/02/25192225/cat.jpg' #Cat
#img_url = 'http://expertofdreams.com/data_images/zebra/zebra-4.jpg' # Zebra
img_url = 'https://i.ytimg.com/vi/cgHZYsFP2cE/maxresdefault.jpg' # G-wagon

def load_and_preprocess_img():
    response = requests.get(img_url)
    img_pil = Image.open(io.BytesIO(response.content))

    normalize = transforms.Normalize(
        mean = [0.4914, 0.4822, 0.4465],
        std = [0.2023, 0.1994, 0.2010]
        )
    preprocess_and_normalize = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize
        ])

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        ])

    img_normalized = preprocess_and_normalize(img_pil)
    img_normalized.unsqueeze_(0)

    img = preprocess(img_pil)
    img.unsqueeze_(0)
    #return Variable(img_tensor)
    return img, img_normalized

def plot_filters(img, activation, num_filters):
    original_img = img[0].numpy().transpose(1,2,0) # To fit tensor into imshow(c, h, w)
    n_columns = 6
    n_rows = math.ceil(num_filters / n_columns) + 1
    plt.figure(1, figsize=(10,10))
    plt.subplot(n_rows, n_columns, 1)
    plt.xticks([]) # Remove labels on axis
    plt.yticks([])
    plt.title('Original image')
    plt.imshow(original_img)
    for i in range(num_filters):
        plt.subplot(n_rows, n_columns, i+2)
        plt.title('Filter ' + str(i + 1))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(activation[0,i,:,:], interpolation="nearest", cmap="gray")


def visualize_first_conv_layer_filters_resnet18(image, image_normalized, num_filters):
    activation = models.resnet18(pretrained=True).conv1(image_normalized)
    plot_filters(image, activation.detach().numpy(), num_filters)



def visualize_last_conv_layer_filters_resnet18(image, image_normalized, num_filters):
    resnet18_last_conv = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])

    for param in resnet18_last_conv.parameters():
        param.requires_grad = False

    output = resnet18_last_conv(image_normalized)

    plot_filters(image, output.detach().numpy(), num_filters)

def visualize_weights_first_conv_layer():
    weights = models.resnet18(pretrained=True).conv1.weight.data.numpy()
    weights = weights[0].transpose(1,2,0)
    plt.title('Weights in first conv layer')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(weights)
    
print('Start preprocessing..')
img, img_norm = load_and_preprocess_img()

#Visualize first convolutional layer
visualize_first_conv_layer_filters_resnet18(img, img_norm, 11)
plt.show()

#Visualize last convolutional layer
visualize_last_conv_layer_filters_resnet18(img, img_norm, 11)
plt.show()

#Visualize weights in first convolutional layer
visualize_weights_first_conv_layer()
plt.show()
#Spørsmål - hvordan printe original bilde?
# Filter i første lag ser etter kanter og slikt
# Siste lag er blitt så abstrahert at det programmet nå ekstraherer av features ikke gir mening for oss. 
