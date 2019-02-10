import numpy as np

# Split training data into a training set and validation set to avoid data snooping
def train_validation_split(img_set, label_set, percentage_in_validation_set):
    split_index = int(round(img_set.shape[0] * (1 - percentage_in_validation_set)))

    return img_set[:split_index], label_set[:split_index], img_set[split_index + 1:], label_set[split_index + 1:]


#def make_batch(set_to_split, start_index, end_index):
#    return set_to_split[start_index:end_index]

def anneleaning_learning_rate(iteration, T_value, initial_learning_rate):
    return initial_learning_rate / (1 + iteration/T_value)

# Filter out the images and labels of 2's and 3's for task 2
def filter_out_2_and_3(img_set, label_set):
    img_2_3_set = np.zeros((0,784))
    label_2_3_set = np.zeros(0)
    for i in range(label_set.shape[0]):
    #for i in range(20000):
        if label_set.item(i) == 2:
            img_2_3_set = np.vstack([img_2_3_set, img_set[i]])
            label_2_3_set = np.append(label_2_3_set, 1)
        elif label_set.item(i) == 3:
            img_2_3_set = np.vstack([img_2_3_set, img_set[i]]) 
            label_2_3_set = np.append(label_2_3_set, 0)
    return img_2_3_set, label_2_3_set

# Shuffle data sets to impove learning and avoid data snooping
def shuffle_sets(img_set, label_set):
    shuffle_index = np.arange(label_set.shape[0]) #Må sikkert endre til shape[1] for softmax
    np.random.shuffle(shuffle_index)

    shuffled_img_set = np.zeros(img_set.shape)
    shuffled_label_set = np.zeros(label_set.shape)

    np.put(shuffled_label_set, shuffle_index, label_set)
    np.put(shuffled_img_set, shuffle_index, img_set)
    
    return shuffled_img_set, shuffled_label_set

    
# One hot code the label set for task 3
def one_hot_encoding(label_set):
    number_of_labels = label_set.shape[0]
    one_hot_label_set = np.zeros((number_of_labels, 10))
    #print(one_hot_label_set.shape)
    one_hot_label_set[np.arange(number_of_labels), label_set] = 1

    return one_hot_label_set

