import os
#import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
#from dataloaders import load_cifar10
#from utils import to_cuda, compute_loss_and_accuracy

class Task2Model1(nn.Module):
    def __init__(self, image_channels, num_classes):
        super(self).__init__()

    # Feature extractor layers
    self.feature_extractor = nn.Sequential(
            nn.Conv2d(                          # Conv 0
                in_channels=image_channels,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(out_channels_in_prev_conv),
            nn.ReLU(),
            nn.Conv2d(                          # Conv 1
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(out_channels_in_prev_conv),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(                          # Conv 2
                in_channels=image_channels,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(out_channels_in_prev_conv),
            nn.ReLU(),
            nn.Conv2d(                          # Conv 3
                in_channels=128,
                out_channels=256, 
#Flere channels gjør at man kan lete etter flere features. Ved å ha flere lag leter man etter features i de allerede definerte featuresene - bedre med mange lag og færre channels enn få lag med mange channels --> det tar lang tid å trene så mange vekter det blir ved å ha mange channels. 
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(out_channels_in_prev_conv),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            '''nn.Conv2d(
                in_channels=image_channels,
                out_channels=conv_1_num_filters,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(out_channels_in_prev_conv),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=conv_1_num_filters,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(out_channels_in_prev_conv),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            '''
            )
            

        self.num_output_features = 265*4*4 #channels*width*height

        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=num_classes),
            )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """

        # Run image through convolutional layers
        x = self.feature_extractor(x)
        #print("x shape" + str(x.shape))
        # Reshape our input to (batch_size, num_output_features)
        x = x.view(-1, self.num_output_features)
        # Forward pass through the fully-connected layers.
        x = self.classifier(x)

        return x
