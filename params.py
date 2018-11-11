import os

#### data ####################################################
# path
KITTI_DIR = "/home/kouichi/Data/kitti_data_npy"
TRAIN_DATA = os.path.join(KITTI_DIR, "X_train.npy")
VAL_DATA = os.path.join(KITTI_DIR, "X_val.npy")
TEST_DATA = os.path.join(KITTI_DIR, "X_test.npy")
# image shape
img_shape = (128, 160, 3)


#### parameters to prepare data ##############################
# the number of frames per one sequence
num_frames = 10 
# the number of videos which consists of 'num_frames' frames
# the total number of frames loaded == 'num_frames' * 'num_sequences'
# if None, all data is loaded
num_sequences = None
# the number of batches fed into the model at each step
batch_size = 10


#### training setting ########################################
# the number of iterations in training
# epoch == 'num_iterations'/('num_sequences'/'batch_size') 
num_iterations = 50000
# learning late
#lr = 0.001 # use default


#### test setting #############################################
# the number of predictions after the input sequence terminated 
pred_length = 5


#### directories to save ###################################
# tensorboad
LOG_DIR = "logs"
VAL_LOG = os.path.join(LOG_DIR, "val")
TRAIN_LOG = os.path.join(LOG_DIR, "train")
# training check point
SAVE_DIR = "saved_model/"
# predicted videos
PRED_DIR = "pred_video"
