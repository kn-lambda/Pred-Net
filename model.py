import tensorflow as tf
import numpy as np

from tensorflow.keras.backend import hard_sigmoid
from tensorflow.keras.layers import UpSampling2D

#### parameter ########################################
# the number of channels in each of the stacked layers
stack_channels = (3, 48, 96, 192)


#### model conponents #################################

class ConvLSTM(object):
    """imprementation of Convolutional LSTM 
    referring to
        https://github.com/takyamamoto/PredNet_Chainer/blob/master/network.py
        https://github.com/joisino/ConvLSTM/blob/master/network.py
        http://joisino.hatenablog.com/entry/2017/10/27/200000
    
    """
    def __init__(self, out_channels, kernel_size=3):
        
        # convolution settings
        # before fed into the first activation, channel size is enlarged to "out_channels"
        # and in after layers, channel size is not changed
        self.out_channels = out_channels
        self.kernel_size = kernel_size
       
        # to the first activation
        self.Wxc = tf.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding="same")
        self.Whc = tf.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding="same", use_bias=False)
        # to input gate
        self.Wxi = tf.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding="same")
        self.Whi = tf.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding="same", use_bias=False)
        # to forget gate
        self.Wxf = tf.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding="same")
        self.Whf = tf.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding="same", use_bias=False)
        # to  output gate
        self.Wxo = tf.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding="same")
        self.Who = tf.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding="same", use_bias=False)

        # peepholes
        # initialized when used for the first time
        self.Wci = None
        self.Wcf = None
        self.Wco = None
        
        # reset internal states
        self.reset_state()
        
        
    def reset_state(self, c=None, h=None):
        self.c = c
        self.h = h
        
        
    def _initialize_peephole(self, height, width):
        shape = (height, width, self.out_channels)
        self.Wci = tf.Variable(tf.zeros(shape))
        self.Wcf = tf.Variable(tf.zeros(shape))
        self.Wco = tf.Variable(tf.zeros(shape))
        
        
    def _initialize_state(self, x):
        # The following calculation is not elegant.
        # To initialize 'c', 'h', their 4-dim shape is required, 
        # especially, the batch-size must be given which may be unknown when constructing the graph.
        # To avoid explicitly using batch-size, abstract input tenor 'x' is fed into the convolution
        # until we get the desired shape.
        tmp = self.Wxc(x)
        self.c = tf.zeros_like(tmp)
        self.h = tf.zeros_like(tmp)
        
        
    def __call__(self, x):
        """one-step execution of convLSTM
        Args:
            x : 4-dim (batch_size, height, width, channel) tensor
        """
        assert len(x.shape) == 4, "the dimension of the input tensor must be {}, but {}.".format(4, len(x.shape))
        
        if self.Wci is None:
            self._initialize_peephole(x.shape[1], x.shape[2])
            
        if self.c is None:
            self._initialize_state(x)

        ig = hard_sigmoid(self.Wxi(x) + self.Whi(self.h) + self.c * self.Wci) #input gate
        fg = hard_sigmoid(self.Wxf(x) + self.Whf(self.h) + self.c * self.Wcf) #forget gate
        new_c = fg * self.c + ig * tf.tanh(self.Wxc(x) + self.Whc(self.h))
        og = hard_sigmoid(self.Wxo(x) + self.Who(self.h) + new_c * self.Wco) #output gate
        new_h = og * tf.tanh(new_c)
        
        self.c = new_c
        self.h = new_h
        
        return new_h


class RepBlock(object):
    """Representation-Neuron in PredNet
    """
    def __init__(self, out_channels):
        self.convlstm = ConvLSTM(out_channels=out_channels)
        self.up_sampler = UpSampling2D(size=(2, 2)) # the size of the image is doubled
        
        
    def reset_state(self):
        self.convlstm.reset_state()
        
        
    def __call__(self, before_R, before_E, above_R=None):
        """
        Args: 
            before_R : R[t-1, l] 
            before_E : E[t-1, l]
            above_R  : R[t, l+1]
        Returns:
            R : R[t, l]
        << t : time, l : layer >>
        """
        
        if above_R is not None:
            up_R = self.up_sampler(above_R)
            input_lstm = tf.concat([before_R, before_E, up_R], axis=3) # concat within channel dim
        else:
            input_lstm = tf.concat([before_R, before_E], axis=3)
            
        R = self.convlstm(input_lstm)
        return R


class ErrBlock(object):
    """Module to make prediction, and extract error
    """
    def __init__(self, out_channels, kernel_size=3, pixcel_max=1.0, is_bottom=False):
        
        self.pixcel_max = pixcel_max
        self.is_bottom = is_bottom
        self.down_sampler = tf.layers.MaxPooling2D(pool_size=2, strides=2, padding="same") # 1/2 down sampling
        self.conv_pred = tf.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding="same")  # to predict
        
        if is_bottom == False: 
            # convolution to extract target from the below layer
            self.conv_target = tf.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding="same")
    
    
    def __call__(self, below_E, R):
        """
        Args: 
            below_E : E[t, l-1] (l>0), input-image (l=0)
            R : R[t, l]
        Returns:
            E : E[t, l]
            A_pred : hatA[t, l]
        << t : time, l : layer >>
        """
        
        if self.is_bottom == False:
            A_target = self.down_sampler(tf.nn.relu(self.conv_target(below_E)))
            A_pred = tf.nn.relu(self.conv_pred(R))
        else:
            A_target = below_E
            # at the bottom layer, A_pred is interpreted as the prediction for the next frame
            A_pred = tf.clip_by_value(tf.nn.relu(self.conv_pred(R)), 0.0, self.pixcel_max)
            
        E = tf.concat([tf.nn.relu(A_pred - A_target), tf.nn.relu(A_target - A_pred)], axis=3)
        
        return E, A_pred


class PredNet(object):
    """Pred Net
    """
    def __init__(self, stack_channels=stack_channels):
        """
        Args:
            stack_chanels : touple of integers representing the number of channels of each layer
                            (first layer, second layer, thier layer, ...)
        """
        
        self.stack_channels = stack_channels
        self.num_layers = len(stack_channels)
        
        for l in range(self.num_layers):
            setattr(self, "R_block"+str(l), RepBlock(out_channels=stack_channels[l]))
            
            if l == 0:
                setattr(self, "E_block"+str(l), ErrBlock(stack_channels[l], is_bottom=True))
            else:
                setattr(self, "E_block"+str(l), ErrBlock(stack_channels[l]))
        
        self.reset_state()
                
                
    def reset_state(self):
        
        self.stack_E = None # value of 'E' at each layer
        self.stack_R = None # value of 'R' at each layer
        
        
    def _one_step(self, x):
        """one time-step execution which follows
           1. from top to bottom, 'R's are updated
           2. predict the next frame, and the next input-data is fed
           3. from bottom to top, 'E's are updated
        Args:
            x : 4-dim (batch_size, height, width, num_channels) tensor
        """
        assert len(x.shape) == 4, "the dimension of the input tensor must be {}, but {}.".format(4, len(x.shape))
        
        # initialize 'R's and 'E's as 0 when started
        if self.stack_E is None:
            
            self.stack_E = []
            self.stack_R = []
            
            # the following calculations are messy
            # as in the case of implementing convLSTM
            tmp_A = tf.zeros_like(x, tf.float32)
            
            for l in range(self.num_layers):
                # calculation to double the number of channels
                chx2 = tf.layers.Conv2D(filters=2*self.stack_channels[l], kernel_size=1, trainable=False, kernel_initializer=tf.zeros_initializer())
                # E.shape == 2 * A.shape
                tmp_E = chx2(tmp_A)
                self.stack_E.append(tf.zeros_like(tmp_E))
                # R.shape == A.shape
                self.stack_R.append(tf.zeros_like(tmp_A))
                
                getattr(self, "R_block"+str(l)).reset_state()
                
                if l != self.num_layers - 1:
                    # calculation to extract the shape in the above layer
                    # 2D-shape is reduced to halve while channel size is increased
                    goup = tf.layers.Conv2D(filters=self.stack_channels[l+1], kernel_size=2, strides=2, trainable=False, kernel_initializer=tf.zeros_initializer())
                    # A[l].shape -> A[l+1].shape
                    tmp_A = goup(tmp_A)
                    
        # update R-block from top to bottom
        for l in reversed(range(self.num_layers)):
            if l != self.num_layers-1:
                new_R = getattr(self, "R_block"+str(l))(self.stack_R[l], self.stack_E[l], self.stack_R[l+1])
            else:
                new_R = getattr(self, "R_block"+str(l))(self.stack_R[l], self.stack_E[l])
            self.stack_R[l] = new_R
            
        # update E-block from bottom to top
        for l in range(self.num_layers):
            if l != 0:
                new_E, _ = getattr(self, "E_block"+str(l))(self.stack_E[l-1], self.stack_R[l])
            else:
                new_E, pred = getattr(self, "E_block"+str(l))(x, self.stack_R[l])     
            
            self.stack_E[l] = new_E

        tmp_loss = tf.reduce_mean(self.stack_E[0]) # loss for this time step
        
        return tmp_loss, pred
    
    
    def __call__(self, X, pred_length=0):
        """execution for time-sequences 
        Args:
            X : 5-dim (batch_size, time_length, height, width, num_channels) tensor
            pred_length (int) : the number of frames predicton are repeated, after the input sequence is terminated
        """
        assert len(X.shape) == 5, "the dimension of the input tensor must be {}, but {}.".format(5, len(X.shape))
        
        time_length = int(X.shape[1])
        
        loss = 0
        pred_list = [] # shape : (time, batch, height, width, channel)
        self.reset_state()
        
        for t in range(time_length + pred_length):
            
            if t < time_length:
                x_t = X[:, t, :, :, :]
            else:
                x_t = pred_list[-1]
                
            loss_t, pred_t = self._one_step(x_t)    
            pred_list.append(pred_t)
            
            # skip loss at t=0
            if t > 0 and t < time_length:
                # accumulate loss
                loss = loss + loss_t
            
        loss = loss / (time_length - 1)
        
        return loss, pred_list
