import mnist
import numpy as np
import math
import matplotlib.pyplot as plt
import helper_functions as hf

# Load MNIST data set
mnist.init()
img_train, label_train, img_test, label_test = mnist.load()

#One hot code labeles
label_train = hf.one_hot_encoding(label_train)
label_test = hf.one_hot_encoding(label_test)

# Normalize input
img_train = img_train / 255
img_test = img_test / 255

# Sets initial weights values
weights = np.random.uniform(-1,1, ((img_train.shape[1], 10)))
#print("Weights shape" + str(weights.shape))

# Tacking variables
train_loss = []
validation_loss = []
test_loss = []
training_iterations = []
last_three_weights = [0,0,0]

def training_loop(batch_size, epochs, initial_learning_rate, regulization_lambda):
    global weights, img_train, label_train
    iteration = 0
    number_of_batches_in_epoch = label_train.shape[0] // batch_size
    print("Number of batches in epoch: " + str(number_of_batches_in_epoch))
    T_value = epochs

    img_set, label_set, img_validation_set, label_validation_set = hf.train_validation_split(img_train, label_train, 0.1)
    for epoch in range(epochs):
        # Shuffle train and validation set 
        #if iteration % 1000 == 0:
        #print("Shuffle..")
        #shuffled_img_set, shuffled_label_set = hf.shuffle_sets(img_train, label_train)
        #print(shuffled_label_set.shape)


        for i in range(number_of_batches_in_epoch):
            iteration += 1

            img_batch = img_set[i*batch_size:(i+1)*batch_size]

            label_batch = label_set[i*batch_size:(i+1)*batch_size]
            if img_batch.shape[0] == 0 or label_batch.shape[0] == 0: continue

            outputs = forward_pass_softmax(img_batch, weights, batch_size)
            learning_rate = hf.anneleaning_learning_rate(iteration, T_value, initial_learning_rate)
            weights = gradient_decent_softmax(img_batch, label_batch, outputs, weights, learning_rate, regulization_lambda, batch_size)

            last_three_weights[iteration % 3] = weights
        

        # Compute losses
        train_output = forward_pass_softmax(img_set, weights, img_set.shape[0])
        train_loss.append(loss_function_softmax(label_set, train_output))
        training_iterations.append(iteration)

        validation_outputs = forward_pass_softmax(img_validation_set, weights, img_validation_set.shape[0])
        validation_loss.append(loss_function_softmax(label_validation_set, validation_outputs))

        test_outputs = forward_pass_softmax(img_test, weights, img_test.shape[0])
        test_loss.append(loss_function_softmax(label_test, test_outputs))

        #Early stopping
        if len(validation_loss) > 4:

            if ( (validation_loss[-1] > validation_loss[-2]) and (validation_loss[-3] > validation_loss[-2]) and (validation_loss[-4] > validation_loss[-3])):
                print("Number of epochs " + str(epoch))
                break

    return last_three_weights[(iteration - 2) % 3]

def gradient_decent_softmax(x, targets, outputs, w, learning_rate, regulization_lambda, batch_size):
    if outputs.shape == targets.shape: 
        dw = - x.T.dot(targets - outputs) 
        dw = dw / batch_size + regulization_lambda*2*w
        assert dw.shape == w.shape, "dw and weights vector of different shape"
        w = w - learning_rate*dw

    return w

def loss_function_softmax(targets, outputs): 
    return - np.sum(targets*np.log(outputs))# , axis = 0)

def forward_pass_softmax(x, w, batch_size):
    y = np.zeros(10)
    index = np.arange(batch_size)
    a = np.zeros((batch_size,10))
    for i in range(batch_size):
        try: 
            np.put(a, i, w.T*x[i])
        except IndexError:
            np.put(a,i, np.zeros((batch_size,10)) )
    return (np.exp(a)) / (np.sum(np.exp(a), axis=0))

# Run training
print("Begins traning..")
initial_batch_size = 30
initial_number_of_epochs = 200
initial_learning_rate = 0.001
initial_regulazation_lambda = 0.01

weights = training_loop(initial_batch_size, initial_number_of_epochs, initial_learning_rate, initial_regulazation_lambda)

# Plot results

print("Len train_loss " + str(len(train_loss)))
print("Len val_loss " + str(len(validation_loss)))
print("Training iterations " + str(training_iterations[-1]))
print("Validation loss = " + str(validation_loss[-1]))
print("Training loss = " + str(train_loss[-1]))
print("Test loss = " + str(test_loss[-1]))

plt.figure(figsize=(12, 8))
#plt.ylim([0, 1])
plt.title("Computed losses for batch size " + str(initial_batch_size) + ", initial learnings rate "+ str(initial_learning_rate) + " and lambda  " + str(initial_regulazation_lambda) )
plt.xlabel("Training steps")
plt.ylabel("Loss")
plt.plot(training_iterations, train_loss, label="Training loss")
plt.plot(training_iterations, validation_loss, label="Validation loss")
plt.plot(training_iterations, test_loss, label="Test loss")
plt.legend() # Shows graph labels
plt.show()

