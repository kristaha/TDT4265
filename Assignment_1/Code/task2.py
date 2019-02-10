import mnist
import numpy as np
import math
import matplotlib.pyplot as plt
import helper_functions as hf

# Load MNIST data set
mnist.init()
img_train, label_train, img_test, label_test = mnist.load()

print("Begin splitting")

# Normalize input
img_train = img_train / 255
img_test = img_test / 255

# Filter out 2's and 3's
img_train, label_train = hf.filter_out_2_and_3(img_train, label_train)
img_test, label_test = hf.filter_out_2_and_3(img_test, label_test)

# Sets initial weights values
weights = np.random.uniform(-1,1, ((img_train.shape[1]))).T

# Tacking variables
train_loss = []
validation_loss = []
test_loss = []
training_iterations = []
last_three_weights = [0,0,0]
accuracy_training_set = []
accuracy_validation_set = []
accuracy_test_set = []
length_of_weight_vector = []



def training_loop(batch_size, epochs, initial_learning_rate, regulization_lambda):
    global weights, img_train, label_train
    global accuracy_training_set, accuracy_validation_set, accuracy_test_set, length_of_weight_vector
    iteration = 0
    number_of_batches_in_epoch = label_train.shape[0] // batch_size
    print("Number of batches in epoch: " + str(number_of_batches_in_epoch))
    T_value = epochs
    
    #Splits data sets into training and validation set.
    img_set, label_set, img_validation_set, label_validation_set = hf.train_validation_split(img_train, label_train, 0.1)
    for epoch in range(epochs):
        #Shuffle training set 
        #shuffled_img_set, shuffled_label_set = hf.shuffle_sets(img_train, label_train)

        for i in range(number_of_batches_in_epoch):
            iteration += 1

            img_batch = img_set[i*batch_size:(i+1)*batch_size]

            label_batch = label_set[i*batch_size:(i+1)*batch_size]
            label_batch = label_batch.T 

            outputs = forward_pass(img_batch, weights)
            
            learning_rate = hf.anneleaning_learning_rate(iteration, T_value, initial_learning_rate)
            weights = gradient_decent_logistic_regression(img_batch, label_batch, outputs, weights, learning_rate, regulization_lambda, batch_size)

            last_three_weights[iteration % 3] = weights
        
        # Compute losses
        if (epoch % 1 == 0):
            train_output = forward_pass(img_set, weights)
            train_loss.append(loss_function_logistic_regression(label_set, train_output))
            training_iterations.append(iteration)
            accuracy_training_set.append(accuracy(label_set, train_output))

            validation_outputs = forward_pass(img_validation_set, weights)
            validation_loss.append(loss_function_logistic_regression(label_validation_set, validation_outputs))
            accuracy_validation_set.append(accuracy(label_validation_set, validation_outputs))

            test_outputs = forward_pass(img_test, weights)
            test_loss.append(loss_function_logistic_regression(label_test, test_outputs))
            accuracy_test_set.append(accuracy(label_test, test_outputs))

            length_of_weight_vector.append(np.sqrt(weights.dot(weights)))

        #Early stopping
        if len(validation_loss) > 4:

            if ( (validation_loss[-1] > validation_loss[-2]) and (validation_loss[-3] > validation_loss[-2]) and (validation_loss[-4] > validation_loss[-3])):
                print("Number of epochs " + str(epoch))
                break

    return last_three_weights[(iteration - 2) % 3]

def gradient_decent_logistic_regression(x, targets, outputs, w, learning_rate, regulization_lambda, batch_size): 
    assert outputs.shape == targets.shape, "Outputs are not of same shape as targets"
    dw = - np.expand_dims((targets - outputs), axis=1)*x #dw.shape = (100,784)
    if dw.shape == (batch_size,784):
        dw = dw.mean(axis=0) 
        dw = dw + regulization_lambda*2*w # dE(w)/dw + dC(w)/dW Gradient descent with  L2 regulization 
        assert dw.shape == w.shape, "dw and weights vector of different shape"
        w = w - learning_rate*dw
    return w

def loss_function_logistic_regression(targets, outputs):
    error = np.zeros((targets.shape[0]))#.T
    error =  ( (targets*np.log(outputs)) + ((1-targets)*(np.log(1-outputs)) ))
    return - error.mean()

def accuracy(targets, outputs):
    predictions = outputs >= 0.5
    correct_predictions = (targets == predictions)
    accuracy = correct_predictions.sum() / len(correct_predictions.squeeze())
    return accuracy

def forward_pass(x, w): #Feeds an input x through the single layer with weights w by finding dot product.
    return 1/( 1+ np.exp(-x.dot(w)))

def lambda_testing():
    global accuracy_training_set, accuracy_validation_set, accuracy_test_set, length_of_weight_vector
    lambdas = [0.01, 0.001, 0.0001]
    lambda_acc = [[], [], []]
    for i in range(len(lambdas)): 
        weights = np.random.uniform(-1,1, ((img_train.shape[1]))).T
        weights = training_loop(initial_batch_size, initial_number_of_epochs, initial_learning_rate, lambdas[i])

        #lambda_acc[i] = accuracy_validation_set
        #accuracy_test_set = []
        #accuracy_validation_set = []
        #accuracy_training_set = []

        lambda_acc[i] = length_of_weight_vector
    return lambda_acc

# Run training
print("Begins traning..")
#weights = training_loop(weights, 100, 200, 0.000001)
initial_batch_size = 30
initial_number_of_epochs = 200
initial_learning_rate = 0.1
initial_regulazation_lambda = 0.0001

weights = training_loop(initial_batch_size, initial_number_of_epochs, initial_learning_rate, initial_regulazation_lambda)
#lamda_training_iterations = []
#acc_sets = lambda_testing()
#print(acc_sets[0])
# Plot results

print("Len train_loss " + str(len(train_loss)))
print("Len val_loss " + str(len(validation_loss)))
print("Training iterations " + str(training_iterations[-1]))
print("Validation loss = " + str(validation_loss[-1])) 
print("Training loss = " + str(train_loss[-1])) 
print("Test loss = " + str(test_loss[-1])) 

plt.figure(figsize=(12, 8))
#plt.ylim([0, 1]) 
plt.title("Magnitude for weights vector for batch size " + str(initial_batch_size) + ", initial learnings rate " + str(initial_learning_rate) + " and lambda  " + str(initial_regulazation_lambda) )
plt.xlabel("Training steps")
plt.ylabel("Magnitude")

#plt.plot(training_iterations, train_loss, label="Training loss")
#plt.plot(training_iterations, validation_loss, label="Validation loss")
#plt.plot(training_iterations, test_loss, label="Test loss")
#plt.plot(training_iterations, accuracy_training_set, label="Training accuracy")
#plt.plot(training_iterations, accuracy_validation_set, label="Validation accuracy")
#plt.plot(training_iterations, accuracy_test_set, label="Test accuracy")
plt.plot(training_iterations, length_of_weight_vector, label="Magnitude")
plt.legend() # Shows graph labels
plt.show()

#Spørsmål til studass : 
#Hvorfor kjører jeg alltid gjennom alle epokene? Det virker som om jeg maks burde kjøre gjennom 5?

