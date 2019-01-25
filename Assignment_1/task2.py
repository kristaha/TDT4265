import mnist
import numpy as np
import math
import matplotlib.pyplot as plt

# Load MNIST data set
mnist.init()
img_train, label_train, img_test, label_test = mnist.load()
img_train, label_train, img_validation, label_validation = train_validation_split(img_train, label_train, 0.1)

# Sets initial weights values
weights = np.zeros((img_train.shape[1], 1)) # Hvorfor ikke bare np.zeros(img_train.shape[0])

# Tacking variables
train_loss = []
validation_loss = []
training_iterations = []
last_three_weights = [0,0,0]

# Run training
weights = training_loop(100, 200, 0.000001)

# Plot results
plt.figure(figsize=(12, 8 ))
plt.ylim([0, 1])
plt.xlabel("Training steps")
plt.ylabel("Loss")
plt.plot(training_iterations, train_loss, label="Training loss")
plt.plot(training_iterations, validation_loss, label="Validation loss")
plt.legend() # Shows graph labels


def training_loop(batch_size, epochs, initial_learning_rate):
    start_index_batch = 0
    validation_freq = 30
    iteration = 0
    number_of_batches_in_epoch = label_train.shape[0] // batch_size

    for epoch in range(epochs):
        #Shuffle train and validation set -- How?

        for i in range(number_of_batches_in_epoch):
            iteration += 1

            img_batch = make_batch(img_train, start_index_batch, start_index_batch + batch_size)
            label_batch = make_batch(label_train, start_index_batch, start_index_batch + batch_size)
            start_index_batch += batch_size

            outputs = forward_pass(img_batch, weights)
            
            learning_rate = anneleaning_learning_rate(iteration, T, initial_learning_rate))
            weights = gradient_decent_logistic_regression(img_batch, label_batch, outputs, weights, learning_rate)

            last_three_weights[iteration % 3] = weights

            #Early stopping
            if i % validation_freq == 0:
                train_output = forward_pass(img_train, weights)
                train_loss.append(loss_function_logistic_regression(label_train, train_output)
                training_iterations.append(iteration)

                validation_outputs = forward_pass(img_validation, weights)
                validation_loss.append(loss_function_logistic_regression(label_validation, validation_outputs))

        if (
            (validation_loss[-1] > validation_loss[-2])
            and (validation_loss[-3] > validation_loss[-2]) 
            and (validation_loss[-4] > validation_loss[-3])
            ):
            
            break

    return last_three_weights[(iteration - 2) % 3]

def gradient_decent_logistic_regression(x, targets, outputs, weights, learning_rate): #x = input vextor. Target = 1 for 2s and 0 for 3s. 
        dw = - (targets - outputs)*x 
        dw = dw.mean(axis=0).reshape(-1,1) #.rechape(nr rows, nr columns). nr rows = -1 => numpy finds row dimention. 
        assert dw.shape == weights.shape, "dw and weights vector of different shape"  
        weights = weights - learning_rate*dw
    return weights

def anneleaning_learning_rate(iteration, T, initial_learning_rate):
    return initial_learning_rate / (1 + iteration/T)

def loss_function_logistic_regression(targets, outputs):
    error =  ( (targets*math.log(outputs)) + ((1-targets)*(math.log(1-outputs)) ))
    return - error.mean()


def forward_pass(x, w): #Feeds an input x through the single layer with weights w by finding dot product. 
    return x.dot(w)

def filter_out_2_and_3(img_set, label_set):
    assert len(img_set) == (label_set), "img set lenght is not the same as label set lenght" 

    for i in range(len(img_set)):
        if not (label_set[i] == 2 or label_set[i] == 3):
            img_set.remove(i)
            label_set.remove(i)
        else if label_set[i] == 2:
            label_set[i] = 1:
        else if label_set[i] == 3:
            label_set[i] = 0

    return img_set, label_set

def train_validation_split(img_set, label_set, percentage_in_validation_set):
    split_index = int(round(img_set.shape[0]*(1-percentage_in_validation_set)))

    return img_set[:split_index], label_set[:split_index],img_set[split_index + 1:], label_set[split_index + 1:]

    
def make_batch(set_to_split, start_index, end_index):
    return set_to_split[start_index:end_index]


