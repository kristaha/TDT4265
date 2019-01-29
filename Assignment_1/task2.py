import mnist
import numpy as np
import math
import matplotlib.pyplot as plt
import helper_functions as hf

# Load MNIST data set
mnist.init()
img_train, label_train, img_test, label_test = mnist.load()
print("Begin splitting")
img_train, label_train = hf.filter_out_2_and_3(img_train, label_train)
img_train, label_train, img_validation, label_validation = hf.train_validation_split(img_train, label_train, 0.1)

# Sets initial weights values
weights = np.zeros((img_train.shape[1])).T

# Tacking variables
train_loss = []
validation_loss = []
training_iterations = []
last_three_weights = [0,0,0]



def training_loop(batch_size, epochs, initial_learning_rate, regulization_lambda):
    global weights
    iteration = 0
    number_of_batches_in_epoch = label_train.shape[0] // batch_size
    print("Number of Batches: " + str(number_of_batches_in_epoch))
    validation_freq = 10 #number_of_batches_in_epoch // 10
    T_value = 1

    for epoch in range(epochs):
        # Shuffle train and validation set -- How?
        for i in range(number_of_batches_in_epoch):
            iteration += 1

            img_batch = img_train[i*batch_size:(i+1)*batch_size]

            label_batch = label_train[i*batch_size:(i+1)*batch_size]
            label_batch = label_batch.T

            outputs = forward_pass(img_batch, weights)
            
            learning_rate = initial_learning_rate ## BYTT UT
            #learning_rate = anneleaning_learning_rate(iteration, T_value, initial_learning_rate)
            weights = gradient_decent_logistic_regression(img_batch, label_batch, outputs, weights, learning_rate)
            weights = regulization(weights, regulization_lambda)

            last_three_weights[iteration % 3] = weights

            #Early stopping
            if i % validation_freq == 0:
                train_output = forward_pass(img_train, weights)
                train_loss.append(loss_function_logistic_regression(label_train, train_output))
                training_iterations.append(iteration)

                validation_outputs = forward_pass(img_validation, weights)
                validation_loss.append(loss_function_logistic_regression(label_validation, validation_outputs))

        if len(validation_loss) > 4:

            if ( (validation_loss[-1] > validation_loss[-2]) and (validation_loss[-3] > validation_loss[-2]) and (validation_loss[-4] > validation_loss[-3])):
                print(epoch)
                break

            

    return last_three_weights[(iteration - 2) % 3]

def gradient_decent_logistic_regression(x, targets, outputs, w, learning_rate): 
    assert outputs.shape == targets.shape, "Outputs are not of same shape as targets"
    dw = - (targets - outputs)*(x.T)
    #print("DW shape")
    #print(dw.shape)
    dw = dw.mean(axis=1) 
    #print("DW shape after mean")
    #print(dw.shape)
    assert dw.shape == w.shape, "dw and weights vector of different shape"
    w = w - learning_rate*dw
    return w

def anneleaning_learning_rate(iteration, T_value, initial_learning_rate):
    return initial_learning_rate / (1 + iteration/T_value)

def loss_function_logistic_regression(targets, outputs):
    error = np.zeros((targets.shape[0]))#.T
    error =  ( (targets*np.log(outputs)) + ((1-targets)*(np.log(1-outputs)) ))
    return - error.mean()


def forward_pass(x, w): #Feeds an input x through the single layer with weights w by finding dot product.
    return x.dot(w)

def regulization(w,lambda_value):
    regulizized_weights = get_error_vector(weights) + lambda_value*get_complexity_penalty_L2(weights)
    return w

def get_error_vector(w):
    return w

def get_complexity_penalty_L2(w):
    return np.sum(w)

# Run training
print("Begins traning..")
#weights = training_loop(weights, 100, 200, 0.000001)
weights = training_loop(100, 200, 0.000001, 0.01)

# Plot results

print("Len train_loss " + str(len(train_loss)))
print("Len val_loss " + str(len(validation_loss)))
print("Training iterations " + str(training_iterations[-1]))

plt.figure(figsize=(12, 8))
plt.ylim([0, 1]) 
plt.xlabel("Training steps")
plt.ylabel("Loss")
plt.plot(training_iterations, train_loss, label="Training loss")
plt.plot(training_iterations, validation_loss, label="Validation loss")
plt.legend() # Shows graph labels
plt.show()

#Spørsmål til studass : 
#Hva skal verdien til T være?
#Hvordan shuffle dataset på en god måte når det er to separate data og labelset? 
