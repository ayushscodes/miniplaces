from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras import metrics

from DataLoader import *
import inspect
import os

import tensorflow as tf
import time

# Dataset Parameters
batch_size = 1024 # How many images are used for one training iteration.
epochs_per_batch = 1
gpu_size = 12 # How many training examples can fit on the GPU?
load_size = 256 # Size of the images on disk.
fine_size = 224 # Size we want the images to be for training.
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842]) # Leave this as is.
num_classes = 100

# Training Parameters
dropout = 0.5 # Dropout, probability to keep units
training_iters = 100000
save_iter = 20
validate_iter = 10
path_save = os.path.join('checkpoints/', 'vgg16_keras_latest.h5')
start_from = ''

if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')

# Construct dataloader
opt_data_train = {
    #'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
}
opt_data_val = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
}

loader_train = DataLoaderDisk(**opt_data_train)
loader_val = DataLoaderDisk(**opt_data_val)

# Define the model.

# weights: 'None' to train on randomly initialized weights, or 'imagenet'
# include_top: whether to include the top 3 fully connected layers of the model
# pooling: 'avg' uses GlobalAveragePooling2D, which returns a 2D tensor
base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
x = base_model.output
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

learning_rate = 0.001
optimizer = Adam(lr=learning_rate)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['top_k_categorical_accuracy', 'categorical_accuracy'])

def convert_labels_categorical(labels_batch):
    labels = np.zeros((batch_size, num_classes))
    for i in range(len(labels_batch)):
        labels[i][int(labels_batch[i])] = 1
    return labels

def save_validation_info(validation_stats, fname='./validation_stats.txt'):
    with open(fname, 'a') as f:
        f.write('%f Loss: %f Top5: %f Top1: %f \n' % (time.time(), validation_stats[0], validation_stats[1], validation_stats[2]))

# Main training loop.
for t_iter in range(training_iters):

    # Train.
    images_batch, labels_batch = loader_train.next_batch(batch_size)
    labels = convert_labels_categorical(labels_batch) # need to make this into a one-hot vector
    model.fit(x=images_batch, y=labels, batch_size=gpu_size, epochs=epochs_per_batch, verbose=1)

    # Show validation stats periodically.
    if t_iter % validate_iter == 0:
        # At the end of this training iteration, test on the validation set.
        images_val, labels_val = loader_val.next_batch(batch_size)
        labels_val_categorical = convert_labels_categorical(labels_val)
        validation_stats = model.evaluate(x=images_val, y=labels_val_categorical, batch_size=gpu_size, verbose=1)
        print('---- Validation results: -----')
        print('Loss: %f Top5: %f Top1: %f' % (validation_stats[0], validation_stats[1], validation_stats[2]))
        print('------------------------------')
        save_validation_info(validation_stats)
    
    # Save periodically.
    if t_iter % save_iter == 0:
        model.save(path_save)

# Do a final save.
model.save(path_save)