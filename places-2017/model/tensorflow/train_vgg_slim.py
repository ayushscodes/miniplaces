# import vgg_slim as vgg
import tensorflow as tf
from DataLoader import *
slim = tf.contrib.slim
from tensorflow.contrib.slim.nets import vgg

# vgg = tf.contrib.slim.nets.vgg

train_log_dir = "./"
if not tf.gfile.Exists(train_log_dir):
  tf.gfile.MakeDirs(train_log_dir)

# Construct dataloader
batch_size = 200
load_size = 256
fine_size = 224
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

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

with tf.Graph().as_default():
  # Set up the data loading:
  images, labels = loader_train.next_batch(batch_size)
  images = tf.cast(images, tf.float32)
  labels = tf.convert_to_tensor(labels, dtype=None)
  images_batch, labels_batch = tf.train.batch([images, labels], batch_size=batch_size)

  # Define the model:
  num_classes = 100
  # inputs: a tensor of size [batch_size, height, width, channels].
  
  predictions = vgg.vgg_16(images_batch, num_classes=num_classes, is_training=True)

  slim.losses.softmax_cross_entropy(predictions[0], labels_batch)

  total_loss = slim.losses.get_total_loss()
  tf.summary.scalar('losses/total_loss', total_loss)

  # Specify the optimization scheme:
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)

  # create_train_op that ensures that when we evaluate it to get the loss,
  # the update_ops are done and the gradient updates are computed.
  train_tensor = slim.learning.create_train_op(total_loss, optimizer)

  # Actually runs training.
  slim.learning.train(train_tensor, train_log_dir)