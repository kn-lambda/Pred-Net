"""
make predicted videos
"""

import tensorflow as tf
import numpy as np
import os, sys, subprocess, gc
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# import model and graph
from params import TEST_DATA, PRED_DIR, SAVE_DIR
from params import img_shape, pred_length, batch_size
from utils import load_npy
from build_graph import X, saver, pred_list


def make_video(org_arr, pred_arr, file_name):
    """make a video from array 
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    ax1.set_title('Actual')
    ax2.set_title('Predicted')

    im_list = []
    org_length = org_arr.shape[0]
    total_length = pred_arr.shape[0]
    
    for t in range(total_length):
        title = fig.text(0.5, 0.85, "t = " + str(t + 1), fontsize = "large")

        if t < org_length:
            im1 = ax1.imshow(org_arr[t])
        else:
            im1 = ax1.imshow(np.zeros(org_arr.shape[1:]))
            
        im2 = ax2.imshow(pred_arr[t])
        im_list.append([im1, im2, title])
        
    ani = animation.ArtistAnimation(fig, im_list, interval=500)

    ani.save(file_name) 
    plt.close(fig)


#### main ###################
# load data
test_arr = load_npy(TEST_DATA)
num_test = test_arr.shape[0]

#video directory
if not os.path.exists(PRED_DIR):
    cmd = "mkdir {}".format(PRED_DIR)
    subprocess.call(cmd.split())

# make prediction
shape = (0, test_arr.shape[1] + pred_length) + img_shape
pred_arr = np.zeros(shape)

with tf.Session() as sess:
    ckpt_state = tf.train.get_checkpoint_state(SAVE_DIR)
    if ckpt_state:
        last_model = ckpt_state.model_checkpoint_path
        saver.restore(sess, last_model)
        
    else:
        sess.run(tf.global_variables_initializer())
        print("No saved models, model is initialized.")
        
    num_loops = num_test // batch_size + 1
    for i in range(num_loops):
        arr_i = test_arr[i * batch_size : (i + 1) * batch_size]
        pred_i = sess.run(pred_list, feed_dict={X:arr_i})
        pred_i = np.array(pred_i).transpose((1,0,2,3,4))
        pred_arr = np.append(pred_arr, pred_i, axis=0)

# rescale arrays as images
pred_arr = (pred_arr * 255).astype(np.uint8)
test_arr = (test_arr * 255).astype(np.uint8)

# make video and save
num_video = test_arr.shape[0]
for i in range(num_video):
    file_name = os.path.join(PRED_DIR, "pred_{}.mp4".format(i))
    make_video(test_arr[i], pred_arr[i], file_name)
    gc.collect()
