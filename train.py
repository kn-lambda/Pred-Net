"""
train and save
"""

import tensorflow as tf
import subprocess, os

from utils import load_npy, iterate_arr
from params import TRAIN_DATA, VAL_DATA, LOG_DIR, SAVE_DIR, VAL_LOG, TRAIN_LOG
from params import num_frames, num_sequences, batch_size, num_iterations

# import model and graph
from build_graph import X, train_op, summary_op
from build_graph import writer_train, writer_val, saver


# load training/validation data
train_arr = load_npy(TRAIN_DATA, num_frames=num_frames, limit=num_sequences)
if num_sequences is None:
    num_sequences = train_arr.shape[0]
train_iter = iterate_arr(train_arr, batch_size=batch_size)
val_arr = load_npy(VAL_DATA)

# clear old log
if os.path.exists(LOG_DIR):
    cmd = "rm -r {}".format(LOG_DIR)
    subprocess.call(cmd.split())

cmd = "mkdir {}".format(LOG_DIR)
subprocess.call(cmd.split())
cmd = "mkdir {}".format(TRAIN_LOG)
subprocess.call(cmd.split())
cmd = "mkdir {}".format(VAL_LOG)
subprocess.call(cmd.split())

# claer old checkpoints
if os.path.exists(SAVE_DIR):
    cmd = "rm -r {}".format(SAVE_DIR)
    subprocess.call(cmd.split())

cmd = "mkdir {}".format(SAVE_DIR)
subprocess.call(cmd.split())

# training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(num_iterations):
        step = i + 1
        epoch = (batch_size * step) // num_sequences
        
        train_batch = next(train_iter)
        sess.run(train_op, feed_dict={X:train_batch})
        
        if step % 10 == 0:
            train_loss = sess.run(summary_op, feed_dict={X:train_batch})
            val_loss = sess.run(summary_op, feed_dict={X:val_arr})

            writer_train.add_summary(train_loss, global_step=step)
            writer_val.add_summary(val_loss, global_step=step)
            
            writer_train.flush()
            writer_val.flush()

            print("epoch {}, total step {} .".format(epoch, step))
            saver.save(sess, SAVE_DIR, write_meta_graph=False, global_step=step)
