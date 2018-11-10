"""
construct graph for train and test
"""

import tensorflow as tf
from model import PredNet
from params import num_frames, img_shape, pred_length
from params import TRAIN_LOG, VAL_LOG


# placefolder for input data
X_shape = (None, num_frames) + img_shape # batch dimension is None
X = tf.placeholder(tf.float32, shape=X_shape)

# Pred Net model
model = PredNet()

# for training and validation
loss, _ = model(X)
train_op = tf.train.AdamOptimizer().minimize(loss)

# for test (or prediction)
_, pred_list = model(X, pred_length=pred_length)

# save check points
saver = tf.train.Saver()

# tensorboard logs
writer_train = tf.summary.FileWriter(TRAIN_LOG)
writer_val = tf.summary.FileWriter(VAL_LOG)
summary_op = tf.summary.scalar("loss", loss)
