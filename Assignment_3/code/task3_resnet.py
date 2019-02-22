import torchvision
from torch import nn

class ResNet18( nn.Module ):
    def __init__(self):
        super ().__init__ ()
        self.model = torchvision.models.resnet18(pretrained = True)
        self.model.fc = nn.Linear(512*4, 10) # No need to apply softmax ,
                                                        # as this is done in nn.CrossEntropyLoss
        for param in self.model.parameters(): # Freeze all parameters
            param.requires_grad = False
        for param in self.model.fc.parameters (): # Unfreeze the last fully - connected layer
            param.requires_grad = True 
        for param in self.model.layer4.parameters (): # Unfreeze the last 5 convolutional layer
            param.requires_grad = True 

    def forward( self, x):
        x = nn.functional.interpolate(x , scale_factor =8)
        x = self.model(x)
        return x
