import numpy as np
def train_validation_split(img_set, label_set, percentage_in_validation_set):
    split_index = int(round(img_set.shape[0] * (1 - percentage_in_validation_set)))

    return img_set[:split_index], label_set[:split_index], img_set[split_index + 1:], label_set[split_index + 1:]


def make_batch(set_to_split, start_index, end_index):
    return set_to_split[start_index:end_index]

def filter_out_2_and_3(img_set, label_set):
    img_2_3_set = np.zeros((0,784))
    label_2_3_set = np.zeros(0)
    #for i in range(label_set.shape[0]):
    for i in range(3000):
        if label_set.item(i) == 2:
            img_2_3_set = np.vstack([img_2_3_set, img_set[i]])
            label_2_3_set = np.append(label_2_3_set, 1)
        elif label_set.item(i) == 3:
            img_2_3_set = np.vstack([img_2_3_set, img_set[i]]) 
            label_2_3_set = np.append(label_2_3_set, 0)
    return img_2_3_set, label_2_3_set
