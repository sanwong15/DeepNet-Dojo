'''
This file describe a Capsule Layer
Reference: https://github.com/naturomics/CapsNet-Tensorflow

San Wong

'''


import numpy as np
import tensorflow as tf

from config import cfg
from utils import reduce_sum
from utils import softmax


epsilon = 1e-9

class CapsLayer(object):
    '''
    Capsule Layer
    Argument Input:
    (4D Tensor) 
    (1) num_outputs: integer (i.e: Number of capsule in this layer)
    (2) vec_len: integer (i.e: Length of the output vector of this layer)
    (3) layer_type: str (i.e: 'FC' or 'CONV')
    (4) with_routing: boolean (i.e: Allowing this capsule to rout with lower-level capsule layer)
    
    Return:
    (1) 4D Tensor
    '''
    
    def __init__(self, num_outputs, vec_len, layer_type='FC', with_routing=True):
        self.num_outputs = num_outputs
        self.vec_len = vec_len
        self.with_routing = with_routing
        self.layer_type = layer_type
        
    def __call__(self, input, kernel_size=None, stride=None):
        '''
        kernel_size and stride will be useful when the layer is defined as CONV
        '''
        
        if self.layer_type == 'CONV':
            self.kernel_size = kernel_size
            self.stride = stride
            
            #Handle case where routing is FALSE
            if not self.with_routing:
                # Without routing, CONV -> PrimaryCaps Layer
                # Input: [batch_size,20,20,256]
                
                # Check input size (output from the 1st conv layer applied on input image)
                assert input.get_shape() == [cfg.batch_size,20,20,256]
                
                '''conv2d(
                            inputs,
                            num_outputs,
                            kernel_size,
                            stride=1,
                            padding='SAME',
                            data_format=None,
                            rate=1,
                            activation_fn=tf.nn.relu,
                            normalizer_fn=None,
                            normalizer_params=None,
                            weights_initializer=initializers.xavier_initializer(),
                            weights_regularizer=None,
                            biases_initializer=tf.zeros_initializer(),
                            biases_regularizer=None,
                            reuse=None,
                            variables_collections=None,
                            outputs_collections=None,
                            trainable=True,
                            scope=None
                        )
                '''
                #Conv1
                '''[According to the paper]: Conv1 has 256, 9 × 9 convolution kernels with a stride of 1 and ReLU activation.
                This layer converts pixel intensities to the activities of local feature detectors that are then used as inputs to the primary capsules
                
                (i.e: self.kernal_size = 256*9*9)
                
                num_outputs: number of output capsule 
                ven_len: Length of the output vector
                '''
                capsules = tf.contrib.layers.conv2d(input, self.num_outputs*self.vec_len, self.kernel_size, self.stride, padding="VALID", activation_fn=tf.nn.relu) # Leaving others to default setting

                capsules = tf.reshape(capsules, (cfg.batch_size, -1, self.vec_len, 1))
                # 32 output capsule * 6*6 output => 32*6*6 = 1152
                # From last layer, it has depth of 256. As now we have 32 capsule, which means each
                # capsule has depth of 256/32 = 8. Each capsule is 8-dim in depth.
                # [batch_size, 1152, 8, 1] <- There are 1152 of 8-Dimension Tensor. Output to Primary Cap Layer
                capsules = squash(capsules)
                assert capsules.get_shape() == [cfg.batch_size, 1152, 8, 1] 
                return(capsules)
            
            
            
        if self.layer_type == 'FC': #FC layer refers to the digit Capsule layer where Routing on Agreement take place
            if self.with_routing:
                # DigitCaps (fully connected layer)
                # Reshape input to [batch_size, 1152,1,8,1]
                # Not sure about the following reshape:
                self.input = tf.reshape(input, shape=(cfg.batch_size, -1, 1, input.shape[-2].value, 1))
                
                with tf.variable_scope('routing'):
                    # b_IJ : [batch_size, num_caps_l, num_caps_lPLUS1, 1,1]
                    # b_IJ is a 1 by 1 value which will later become c_IJ
                    b_IJ = tf.constant(np.zeros([cfg.batch_size, input.shape[1].value, self.num_outputs,1,1],dtype=np.float32))
                    capsules = routing(self.input, b_IJ)
                    capsules = tf.squeeze(capsules, axis = 1)
            return(capsules)
        
    def routing(input, b_IJ):
        '''Arg: input_tensor: [batch_size, num_caps_l = 1152, 1, len(u_i)=8, 1]
        
           Return: [batch_size, num_caps_l_plus_one, len(v_j)=16, 1]
           
           u_i represents the vector output of capsule i in the layer l, and
           v_j the vector output of capsule j in the layer l+1.
        '''
        
        # W: [1, num_caps_i, num_caps_j*len_v_j, len_u_j, 1]
        W = tf.get_variable('Weight',shape=(1,1152,160,8,1),dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=cfg.stddev))
        biases = tf.get_variable('bias', shape=(1,1,10,16,1))
        
        # cal u_hat
        '''
        Since tf.matmul is a time-consuming op,
        A better solution is using element-wise multiply, reduce_sum and reshape
        ops instead. Matmul [a, b] x [b, c] is equal to a series ops as
        element-wise multiply [a*c, b] * [a*c, b], reduce_sum at axis=1 and
        reshape to [a, c]
        '''
        
        '''
        tf.tile create a new tensor by replicating input multiples times
        output tensor's i_th dimension has input.dims(i)*multiples[i]
        elements and the value of input are replicated multiples[i] times along
        the i_th dimension
        
        Example: [a b c d] by [2] output [a b c d a b c d]
        '''
        
        # input_tensor: [batch_size, num_caps_l = 1152, 1, len(u_i)=8, 1]
        input = tf.tile(input, [1,1,160,1,1])
        # Validate if input shape 
        assert input.get_shape() == [cfg.batch_size,1152,160,8,1]
        
        u_hat = tf.reduce_sum(W*input, axis=3, keepdims = True) # Element-wise sum
        u_hat = tf.reshape(u_hat, shape=[-1,1152,10,16,1])
        #check size
        assert u_hat.get_shape() == [cfg.batch_size,1152,10,16,1]
        
        # During forward pass, u_hat_stopped == u_hat
        # No update during backprop. no gradient pass either
        u_hat_stopped = tf.stop_gradient(u_hat,name='stop_gradient')
        
        for r_iter in range(cfg.iter_routing):
            with tf.variable('iter_'+str(r_iter)):
                #[batch_size, 1152, 10, 1, 1]
                c_IJ = softmax(b_IJ, axis=2)
                
                if r_iter == cfg.iter_routing -1: #last iteration: we use u_hat
                    s_J = tf.multiply(c_IJ,u_hat)
                    s_J = reduce_sum(s_J,axis=1,keepdims=True) + biases
                    assert s_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]
                    
                    v_J = squash(s_J)
                    assert v_J.get_shape() == [cfg.batch_size, 1,10,16,1]
                elif r_iter < cfg.iter_routing-1:
                    s_J = tf.multiply(c_IJ,u_hat_stopped)
                    s_J = reduce_sum(s_J,axis=1,keepdims=True)+biases
                    v_J = squash(s_J)
                    
                    
                    # Reshape and tile v_J from [batch_size, 1, 10, 16, 1]
                    # to match with u_hat_stopped: [batch_size, 1152, 10,16, 1]
                    # b_IJ += u_hat_stopped^T * v_J
                    v_J_tiled = tf.tile(v_J, [1,1152,1,1,1])
                    
                    # v_J_tiled: [batch_size, 1152, 10,16, 1]
                    # u_hat_stopped: [batch_size,1152,10,16,1]
                    u_produce_v = reduce_sum(u_hat_stopped*v_J_tiled, axis=3, keepdims=True)
                    assert u_produce_v.get_shape() == [cfg.batch_size,1152,10,1,1]
                    
                    # b_IJ +=
                    b_IJ += u_produce_v
                    
                    
        return(v_J)
    
    def squash(vector):
        '''
        Input: tensor with shape: [batch_size, 1, num_caps, vec_len, 1]
        
        Return: same shape. squashed in vec_len dimension
        '''
        
        vec_squashed_norm = reduce_sum(tf.square(vector),-2,keepdims=True)
        scalar_factor = vec_squashed_norm/(1+vec_squashed_norm)/tf.sqrt(vec_squashed_norm + epsilon)
        vec_squashed = scalar_factor*vector
        
        return(vec_squashed)