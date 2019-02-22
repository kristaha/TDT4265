import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from dataloaders import load_cifar10
from utils import to_cuda, compute_loss_and_accuracy
from leNet_model import LeNetModel
from task2_model_1 import Task2Model1
from task2_model_2 import Task2Model2
from task3_resnet import ResNet18

class Trainer:

    def __init__(self, batch_size, learning_rate, network, optimizer):
        """
        Initialize our trainer class.
        Set hyperparameters, architecture, tracking variables etc.
        """
        # Define hyperparameters
        self.epochs = 100
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stop_count = 4

        # Architecture

        # Since we are doing multi-class classification, we use the CrossEntropyLoss
        self.loss_criterion = nn.CrossEntropyLoss()
        # Initialize the mode
        if network == 0:
            self.model = Task2Model1(image_channels=3, num_classes=10)
        elif network == 1:
            self.model = ResNet18()

        #self.model = LeNetModel(image_channels=3, num_classes=10)
        #self.model = Task2Model1(image_channels=3, num_classes=10)
        #self.model = Task2Model2(image_channels=3, num_classes=10)
        #self.model = ResNet18()

        # Initialize Xavier weights
        def weights_init(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.model.apply(weights_init)
        # Transfer model to GPU VRAM, if possible.
        self.model = to_cuda(self.model)

        # Define our optimizer. SGD = Stochastich Gradient Descent
        if optimizer == 0:
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             self.learning_rate)
        elif optimizer == 1:
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)

        # Load our dataset
        self.dataloader_train, self.dataloader_val, self.dataloader_test = load_cifar10(self.batch_size)

        self.validation_check = len(self.dataloader_train) // 2

        # Tracking variables
        self.VALIDATION_LOSS = []
        self.TEST_LOSS = []
        self.TRAIN_LOSS = []
        self.TRAIN_ACC = []
        self.VALIDATION_ACC = []
        self.TEST_ACC = []
        self.number_of_epochs = 0

    def validation_epoch(self):
        """
            Computes the loss/accuracy for all three datasets.
            Train, validation and test.
        """
        self.model.eval()

        # Compute for training set
        train_loss, train_acc = compute_loss_and_accuracy(
            self.dataloader_train, self.model, self.loss_criterion
        )
        self.TRAIN_ACC.append(train_acc)
        self.TRAIN_LOSS.append(train_loss)

        # Compute for validation set
        validation_loss, validation_acc = compute_loss_and_accuracy(
            self.dataloader_val, self.model, self.loss_criterion
        )
        self.VALIDATION_ACC.append(validation_acc)
        self.VALIDATION_LOSS.append(validation_loss)
        print("Current validation loss:", validation_loss, " Accuracy:", validation_acc)
        # Compute for testing set
        test_loss, test_acc = compute_loss_and_accuracy(
            self.dataloader_test, self.model, self.loss_criterion
        )
        self.TEST_ACC.append(test_acc)
        self.TEST_LOSS.append(test_loss)

        self.model.train()

    def should_early_stop(self):
        """
        Checks if validation loss doesn't improve over early_stop_count epochs.
        """
        # Check if we have more than early_stop_count elements in our validation_loss list.
        if len(self.VALIDATION_LOSS) < self.early_stop_count:
            return False
        # We only care about the last [early_stop_count] losses.
        relevant_loss = self.VALIDATION_LOSS[-self.early_stop_count:]
        previous_loss = relevant_loss[0]
        for current_loss in relevant_loss[1:]:
            # If the next loss decrease, early stopping criteria is not met.
            if current_loss < previous_loss:
                return False
            previous_loss = current_loss
        return True


    def train(self):
        """
        Trains the model for [self.epochs] epochs.
        """
        # Track initial loss/accuracy
        self.validation_epoch()
        for epoch in range(self.epochs):
            # Perform a full pass through all the training samples
            self.validation_epoch()
            for batch_it, (X_batch, Y_batch) in enumerate(self.dataloader_train):
                # X_batch is the CIFAR10 images. Shape: [batch_size, 3, 32, 32]
                # Y_batch is the CIFAR10 image label. Shape: [batch_size]
                # Transfer images / labels to GPU VRAM, if possible
                X_batch = to_cuda(X_batch)
                Y_batch = to_cuda(Y_batch)

                # Perform the forward pass
                predictions = self.model(X_batch)
                # Compute the cross entropy loss for the batch
                loss = self.loss_criterion(predictions, Y_batch)

                # Backpropagation
                loss.backward()

                # Gradient descent step
                self.optimizer.step()
                
                # Reset all computed gradients to 0
                self.optimizer.zero_grad()
                 # Compute loss/accuracy for all three datasets.
                if batch_it % self.validation_check == 0:
                    #self.validation_epoch()
                    # Check early stopping criteria.
                    if self.should_early_stop():
                        print("Early stopping.")
                        return
            self.number_of_epochs += 1

    def visualize_first_filter_resnet18():
        image = self.dataloader_train[0]
        activation = torchvision.models.resnet18(pretrained=True).conv1(image)



if __name__ == "__main__":
    #model21 = Trainer(64, 5e-2, 0, 0)
    resnet18 = Trainer(32, 5e-4, 1, 1)

    resnet18.train()
    #model21.train()

    os.makedirs("plots", exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(12, 8))
    plt.title("Cross Entropy Loss")
    plt.plot(model21.VALIDATION_LOSS, label="Validation loss Model 21")
    plt.plot(model21.TRAIN_LOSS, label="Training loss Model 21")
    plt.plot(model21.TEST_LOSS, label="Testing Loss Model 21")
    plt.plot(resnet18.VALIDATION_LOSS, label="Validation loss ResNet18")
    plt.plot(resnet18.TRAIN_LOSS, label="Training loss ResNet18")
    plt.plot(resnet18.TEST_LOSS, label="Testing Loss ResNet18")
    plt.legend()
    plt.savefig(os.path.join("plots", "final_loss.png"))
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.title("Accuracy")
    plt.plot(model21.VALIDATION_ACC, label="Validation Accuracy Model 21")
    plt.plot(model21.TRAIN_ACC, label="Training Accuracy Model 21")
    plt.plot(model21.TEST_ACC, label="Testing Accuracy Model 21")
    plt.plot(resnet18.VALIDATION_ACC, label="Validation Accuracy ResNet18")
    plt.plot(resnet18.TRAIN_ACC, label="Training Accuracy ResNet18")
    plt.plot(resnet18.TEST_ACC, label="Testing Accuracy ResNet18")
    plt.legend()
    plt.savefig(os.path.join("plots", "final_accuracy.png"))
    plt.show()
##Spørsmål til studass 
    # Meningen at det skal ta flere timer å kjøre resnet18?


    #print("Final test accuracy:", trainer.TEST_ACC[-trainer.early_stop_count])
    #print("Final test loss:", trainer.TEST_LOSS[-trainer.early_stop_count])
    #print("Final validation accuracy:", trainer.VALIDATION_ACC[-trainer.early_stop_count])
    #print("Final validation loss:", trainer.VALIDATION_LOSS[-trainer.early_stop_count])
    #print("Final ttaining accuracy:", trainer.TRAIN_ACC[-trainer.early_stop_count])
    #print("Final training loss:", trainer.TRAIN_LOSS[-trainer.early_stop_count])
    #print("Total number of epochs " + str(trainer.number_of_epochs))
