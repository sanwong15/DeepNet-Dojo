'''
This file describe a Capsule Network
Reference: https://github.com/naturomics/CapsNet-Tensorflow

San Wong

'''

import tensorflow as tf

from config import cfg
from utils import get_batch_data
from utils import softmax
from utils import reduce_sum
from capsLayer import CapsLayer


epsilon = 1e-9

class CapsNet(object):
	def __init__(self, is_training = True):
		self.graph = tf.Graph()

		with self.graph.as_default():
			if is_training:
				self.X, self.labels = get_batch_data(cfg.dataset, cfg.batch_size, cfg.num_threads)
				self.Y = tf.one_hot(self.labels, depth = 10, axis = 1, dtype = tf.float32) #depth = 10 for 10 classes

				self.build_arch()
				self.loss()
				self.summary()

				self.global_step = tf.Variable(0,name='global_step',trainable=False)
				self.optimizer = tf.train.AdamOptimizer()
				self.train_op = self.optimizer.minimize(self.total_loss, global_step = self.global_step)

			else:
				#Which is either Testing or Validation
				self.X = tf.placeholder(tf.float32, shape = (cfg.batch_size,28,28,1)) # 28 by 28 pixel and 1 channel
				self.labels = tf.placeholder(tf.int32, shape = (cfg.batch_size, ))
				self.Y = tf.reshape(self.labels, shape = (cfg.batch_size, 10, 1))
				self.build_arch()

		tf.logging.info('Seting up the main structure')

		def build_arch(self):
			with tf.variable_scope('Conv1_layer'):
				# Conv1_layer:
				# Input [batch_size, 20, 20, 256]
				conv1 = tf.contrib.layers.conv2d(self.X, num_outputs=256, kernel_size=9, stride=1, padding='VALID')
				assert conv1.get_shape() == [cfg.batch_size, 20, 20, 256]

			# Primary Cap Layer
			# Output: [batch_size, 6, 6, 32, 8-Dim tensor]
			# i.e: [cfg.batch_size, 1152, 8, 1]
			with tf.variable_scope('PrimaryCaps_layer'):
				primaryCaps = CapsLayer(num_outputs=32, vec_len=8, with_routing=False, layer_type='CONV')
				caps1 = primaryCaps(conv1,kernel_size=9,stride=2)
				assert caps1.get_shape() == [cfg.batch_size,1152,8,1]

			with tf.variable_scope('DigitCaps_layer'):
				digitCaps = CapsLayer(num_outputs=10, vec_len=16,with_routing=True,layer_type='FC')
				self.caps2 = digitCaps(caps1) # Don't understand 

			# REVIEW WHAT's MASKING
		    with tf.variable_scope('Masking'):
				# calculate ||v_c||, then softmax(||v_c||)
				# [batch_size, 10, 16, 1] => [batch_size, 10, 1, 1]
				self.v_length = tf.sqrt(reduce_sum(tf.square(self.caps2),axis=2,keepdims=True)+epsilon)
				self.softmax_v = softmax(self.v_length, axis=1)
				assert self.softmax_v == [cfg.batch_size, 10, 1, 1]

			    # Pick the index with the max softmax val of the 10 caps
			    # [batch_size, 10, 1 ,1] => [batch_size] (index)
				self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
				assert self.argmax_idx.get_shape() == [cfg.batch_size, 1,1]
				self.argmax_idx = tf.reshape(self.argmax_idx, shape=(cfg.batch_size, ))

				# WHAT's MASK WITH Y
				if not cfg.mask_with_y:
					# indexing
					masked_v = []
					for batch_size in range(cfg.batch_size):
						v = self.caps[batch_size][se;f.argmax_idx[batch_size], :]
						masked_v.append(tf.reshape(v,shape=(1,1,16,1)))

					self.masked_v = tf.concat(masked_v, axis=0)
					assert self.masked_v.get_shape() == [cfg.batch_size, 1, 16, 1]
				else:
					# MASK WITH TRUE LABEL
					self.masked_v = tf.multiply(tf.squeeze(self.caps2), tf.reshape(self.Y,(-1,10,1)))
					self.v_length = tf.sqrt(reduce_sum(tf.square(self.caps2),axis=2,keepdims=True)+epsilon)

			with tf.variable_scope('Decoder'):
				# Reconstructe MNIST image with 3 FC layers
				# [batch_size]



