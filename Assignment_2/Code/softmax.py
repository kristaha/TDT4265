import numpy as np
import matplotlib.pyplot as plt
import mnist
import tqdm


def should_early_stop(validation_loss, num_steps=3):
    if len(validation_loss) < num_steps+1:
        return False

    is_increasing = [validation_loss[i] <= validation_loss[i+1] for i in range(-num_steps-1, -1)]
    return sum(is_increasing) == len(is_increasing) 

def train_val_split(X, Y, val_percentage):
  """
    Selects samples from the dataset randomly to be in the validation set. Also, shuffles the train set.
    --
    X: [N, num_features] numpy vector,
    Y: [N, 1] numpy vector
    val_percentage: amount of data to put in validation set
  """
  dataset_size = X.shape[0]
  idx = np.arange(0, dataset_size)
  np.random.shuffle(idx) 
  
  train_size = int(dataset_size*(1-val_percentage))
  idx_train = idx[:train_size]
  idx_val = idx[train_size:]
  X_training, Y_training = X[idx_train], Y[idx_train]
  X_val, Y_val = X[idx_val], Y[idx_val]
  return X_training, Y_training, X_val, Y_val

def shuffle_train_set(X, Y):
  dataset_size = X.shape[0]
  idx = np.arange(0, dataset_size)
  np.random.shuffle(idx) 
  X_training, Y_training = X[idx], Y[idx]
  return X_training, Y_training



def onehot_encode(Y, n_classes=10):
    onehot = np.zeros((Y.shape[0], n_classes))
    onehot[np.arange(0, Y.shape[0]), Y] = 1
    return onehot

def bias_trick(X):
    return np.concatenate((X, np.ones((len(X), 1))), axis=1)

def check_gradient(X, targets, w_j, w_k, epsilon, computed_gradient_hidden_layer, computed_gradient_output_layer):
    print("Checking gradient...")
    dw_j = np.zeros_like(w_j)
    dw_k = np.zeros_like(w_k)
    for i in range(w_j.shape[0]):
        for l in range(w_j.shape[1]):
            new_w_j_1, new_w_j_2 = np.copy(w_j), np.copy(w_j)
            new_w_j_1[i,l] += epsilon
            new_w_j_2[i,l] -= epsilon

            loss1 = cross_entropy_loss(X, targets, new_w_j_1, w_k)
            loss2 = cross_entropy_loss(X, targets, new_w_j_2, w_k)

            dw_j[i,l] = (loss1 - loss2) / (2*epsilon)

    maximum_abosulte_difference = abs(computed_gradient_hidden_layer-dw_j).max()
    assert maximum_abosulte_difference <= epsilon**2, "Absolute error for dw_j was: {}".format(maximum_abosulte_difference)

    for n in range(w_k.shape[0]):
        for m in range(w_k.shape[1]):
            new_w_k_1, new_w_k_2 = np.copy(w_k), np.copy(w_k)
            new_w_k_1[n,m] += epsilon
            new_w_k_2[n,m] -= epsilon

            loss1 = cross_entropy_loss(X, targets, w_j, new_w_k_1)
            loss2 = cross_entropy_loss(X, targets, w_j, new_w_k_2)

            dw_k[n,m] = (loss1 - loss2) / (2*epsilon)
    maximum_abosulte_difference = abs(computed_gradient_output_layer - dw_k).max()
    assert maximum_abosulte_difference <= epsilon**2, "Absolute error  for dw_k was: {}".format(maximum_abosulte_difference)

def softmax(a):
    a_exp = np.exp(a)
    return a_exp / a_exp.sum(axis=1, keepdims=True)

def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))

def sigmoid_derivative(a):
    return sigmoid(a)*(1-sigmoid(a))

def improved_sigmoid(a):
    return 1.7159*np.tanh((2/3)*a)

def improved_sigmoid_derivative(a):
    return (1.7159*(2/3)) * (1-np.tanh((2/3)*a)**2)

def forward_output_layer(X, w):
    a = X.dot(w.T)
    return softmax(a)

def forward_hidden_layer(X, w):
    z = X.dot(w.T)
    #return sigmoid(z)
    return improved_sigmoid(z)

def forward(X, w_j_1, w_j_2, w_k):
    output_hidden_layer_1 = forward_hidden_layer(X, w_j_1)
    output_hidden_layer_2 = forward_hidden_layer(output_hidden_layer_1, w_j_2)
    output = forward_output_layer(output_hidden_layer_2, w_k)
    return output

def get_dropout_index(w, p):
    return np.random.binomial(1, p, w.shape)

def do_dropout(w, drop_index):
    return np.multiply(w, drop_index)

def scale_for_dropout(w, p):
    return p*w

def calculate_accuracy(X, targets, w_j_1, w_j_2, w_k):
    output = forward(X, w_j_1, w_j_2, w_k)
    predictions = output.argmax(axis=1)
    targets = targets.argmax(axis=1)
    return (predictions == targets).mean()

def cross_entropy_loss(X, targets, w_j_1, w_j_2, w_k):
    output = forward(X, w_j_1, w_j_2, w_k)
    assert output.shape == targets.shape 
    log_y = np.log(output)
    cross_entropy = -targets * log_y
    return cross_entropy.mean() # This normalizes both by batch size and number of classes like in gradient descent. 

def gradient_descent(X, targets, w_j_1, w_j_2, w_k,  learning_rate, momentum_rate, 
        prev_dw_j_1, prev_dw_j_2, prev_dw_k, dropout_percent, should_check_gradient):

    normalization_factor = X.shape[0] * targets.shape[1] # batch_size * num_classes

    # Finds weights to drop during dropout
    hidden_layer_1_dropouts = get_dropout_index(w_j_1, dropout_percent)
    hidden_layer_2_dropouts = get_dropout_index(w_j_2, dropout_percent)
    
    # Calculate temporary weights as a result of dropout
    w_j_1_droped = do_dropout(w_j_1, hidden_layer_1_dropouts)
    w_j_2_droped = do_dropout(w_j_2, hidden_layer_2_dropouts)

    outputs = forward(X, w_j_1_droped, w_j_2_droped, w_k)

    # Calculations needed for backpropagation
    a_j_1 = forward_hidden_layer(X, w_j_1_droped)
    z_j_1 = X.dot(w_j_1_droped.T)

    a_j_2 = forward_hidden_layer(a_j_1, w_j_2_droped)
    z_j_2 = a_j_1.dot(w_j_2_droped.T)

    # Delta for output layer
    delta_k = - (targets - outputs) # Shape (64,10) = Batch size, output nodes

    #delta_j = sigmoid_derivative(z_j)*delta_k.dot(w_k) #shape (64,24)
    # Delta for second hidden layer
    delta_j_2 = improved_sigmoid_derivative(z_j_2)*delta_k.dot(w_k) #shape (64,24)
    
    # Delta for first hidden layer
    delta_j_1 = improved_sigmoid_derivative(z_j_1)*delta_j_2.dot(w_j_2_droped) #shape (64,24)

    # Update output layer weights
    dw_k = delta_k.T.dot(a_j_2) #Shape (10, 24)
    dw_k = dw_k / normalization_factor # Normalize gradient equally as loss normalization
    dw_k = learning_rate*dw_k
    assert dw_k.shape == w_k.shape, "dw_k shape was: {}. Expected: {}".format(dw_k.shape, w_k.shape)
    w_k = w_k - dw_k - momentum_rate*prev_dw_k


    # Update hidden layer 2 weights
    dw_j_2 = (delta_j_2.T.dot(a_j_1)) 
    dw_j_2 = dw_j_2 / normalization_factor
    dw_j_2 = learning_rate*dw_j_2
    dw_j_2 = do_dropout(dw_j_2, hidden_layer_2_dropouts)
    assert dw_j_2.shape == w_j_2.shape, "dw_j_2 shape after mean was: {}. Expected: {}".format(dw_j_2.shape, w_j_2.shape)
    w_j_2 = w_j_2 - dw_j_2 - momentum_rate*prev_dw_j_2
    
    # Update hidden layer 1 weights
    dw_j_1 = (delta_j_1.T.dot(X)) 
    dw_j_1 = dw_j_1 / normalization_factor
    dw_j_1 = learning_rate*dw_j_1
    dw_j_1 = do_dropout(dw_j_1, hidden_layer_1_dropouts)
    assert dw_j_1.shape == w_j_1.shape, "dw_j_1 shape after mean was: {}. Expected: {}".format(dw_j_1.shape, w_j_1.shape)
    w_j_1 = w_j_1 - dw_j_1 - momentum_rate*prev_dw_j_1

    if should_check_gradient:
        check_gradient(X, targets, w_j, w_k, 1e-2,  dw_j, dw_k)

    return w_j_1, w_j_2, w_k, dw_j_1, dw_j_2, dw_k

# Read in data
X_train, Y_train, X_test, Y_test = mnist.load()

# Pre-process data
X_train, X_test = (X_train / 127.5) - 1 , (X_test / 127.5) - 1 # Normalize input 
X_train = bias_trick(X_train)
X_test = bias_trick(X_test)
Y_train, Y_test = onehot_encode(Y_train), onehot_encode(Y_test) # One-hot encode targets to fit softmax function. 

X_train, Y_train, X_val, Y_val = train_val_split(X_train, Y_train, 0.1) # Split training set into a training set and validation set


# Hyperparameters
batch_size = 32
learning_rate = 0.9
momentum_rate = 0.9 # Meget gode resultater med 0.1
percent_of_units_to_drop = 0.5
num_batches = X_train.shape[0] // batch_size
should_gradient_check = False
check_step = num_batches // 10
max_epochs = 20
units_in_hidden_layer = 59

# Tracking variables
TRAIN_LOSS = []
TEST_LOSS = []
VAL_LOSS = []
TRAIN_ACC = []
TEST_ACC = []
VAL_ACC = []

def train_loop(x_train, y_train):
    # Initialize weights
    #w_hidden_layer = np.random.uniform(-1, 1, (units_in_hidden_layer, x_train.shape[1]))
    #w_output_layer = np.random.uniform(-1,1, (y_train.shape[1], units_in_hidden_layer))
    
    # Initializing weights from a normal distribution
    w_hidden_layer_1 = np.random.normal(0, 1/np.sqrt(x_train.shape[1]), (units_in_hidden_layer, x_train.shape[1]))
    w_hidden_layer_2 = np.random.normal(0, 1/np.sqrt(units_in_hidden_layer), (units_in_hidden_layer, units_in_hidden_layer))
    w_output_layer = np.random.normal(0, 1/np.sqrt(units_in_hidden_layer), (y_train.shape[1], units_in_hidden_layer))

    # Scaled weights used for testing with droput
    test_w_hidden_layer_1 = np.zeros(w_hidden_layer_1.shape)
    test_w_hidden_layer_2 = np.zeros(w_hidden_layer_2.shape)

    # Variables for storing previous dw needed for implementation of momentum
    last_dw_hidden_layer_1 = np.zeros(w_hidden_layer_1.shape)
    last_dw_hidden_layer_2 = np.zeros(w_hidden_layer_2.shape)
    last_dw_output_layer = np.zeros(w_output_layer.shape)

    should_gradient_check = False

    for e in range(max_epochs): # Epochs
        for i in tqdm.trange(num_batches):

            x_batch = x_train[i*batch_size:(i+1)*batch_size]
            y_batch = y_train[i*batch_size:(i+1)*batch_size]
            
            w_hidden_layer_1, w_hidden_layer_2, w_output_layer, last_dw_hidden_layer_1, last_dw_hidden_layer_2, last_dw_output_layer = gradient_descent(
                    x_batch, y_batch, w_hidden_layer_1, w_hidden_layer_2, w_output_layer,
                    learning_rate, momentum_rate, last_dw_hidden_layer_1, last_dw_hidden_layer_2,
                    last_dw_output_layer, percent_of_units_to_drop, should_gradient_check)
            
            if should_gradient_check:
                should_gradient_check = False

            if i % check_step == 0:
                
                # Loss
                test_w_hidden_layer_1 = scale_for_dropout(w_hidden_layer_1, percent_of_units_to_drop)
                test_w_hidden_layer_2 = scale_for_dropout(w_hidden_layer_2, percent_of_units_to_drop)

                TRAIN_LOSS.append(cross_entropy_loss(X_train, Y_train, test_w_hidden_layer_1,
                    test_w_hidden_layer_2, w_output_layer))
                TEST_LOSS.append(cross_entropy_loss(X_test, Y_test, test_w_hidden_layer_1,
                    test_w_hidden_layer_2, w_output_layer))
                VAL_LOSS.append(cross_entropy_loss(X_val, Y_val, test_w_hidden_layer_1, 
                    test_w_hidden_layer_2, w_output_layer))
                

                TRAIN_ACC.append(calculate_accuracy(X_train, Y_train, test_w_hidden_layer_1,
                    test_w_hidden_layer_2, w_output_layer))
                VAL_ACC.append(calculate_accuracy(X_val, Y_val, test_w_hidden_layer_1, 
                    test_w_hidden_layer_2, w_output_layer))
                TEST_ACC.append(calculate_accuracy(X_test, Y_test, test_w_hidden_layer_1,
                    test_w_hidden_layer_2, w_output_layer))
                if should_early_stop(VAL_LOSS):
                    print(VAL_LOSS[-4:])
                    print(TEST_ACC[-4])
                    print("early stopping.")
                    return w_hidden_layer_1, w_hidden_layer_2, w_output_layer
        # Shuffle training set
        x_train, y_train = shuffle_train_set(x_train, y_train)
        #if e > 4:
         #   should_gradient_check = True
    print(TEST_ACC[-1])
    return w_hidden_layer_1, w_hidden_layer_2, w_output_layer

#'''
w_j1, w_j2, w_k = train_loop(X_train, Y_train)

plt.plot(TRAIN_LOSS, label="Training loss")
plt.plot(TEST_LOSS, label="Testing loss")
plt.plot(VAL_LOSS, label="Validation loss")
plt.ylabel("Cross entropy loss")
plt.xlabel("Training steps")
plt.legend()
plt.ylim([0, 0.07])
plt.show()

plt.clf()
plt.plot(TRAIN_ACC, label="Training accuracy")
plt.plot(TEST_ACC, label="Testing accuracy")
plt.plot(VAL_ACC, label="Validation accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Training steps")
plt.ylim([0.8, 1.0])
plt.legend()
plt.show()

plt.clf()
    
'''w = w[:, :-1] # Remove bias
w = w.reshape(10, 28, 28)
w = np.concatenate(w, axis=0)
plt.imshow(w, cmap="gray")
plt.show()
'''

#Spørsmål til studass:
# Hvordan få samme kompleksitetsnivå som før? Nå har jeg bare lagt på et fully conected hidden_layer som øker kompleksiteten
# Hvilke endringer burde jeg fått i 4B?
# Hvorfor forskjeller i antall epochs før early stopp?
# Hvor mange units bør droppes ved dropout? Jeg får ikke noe bedre resultater.




