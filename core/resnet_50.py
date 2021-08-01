import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D
from tensorflow.python.keras.engine import input_layer
from core.deformable_conv_layer import DeformableConvLayer
def resnet_50(resnet,training=True):
 
  if not training:  # For lyers freeze
      print('Resnet layers are freezed...')
      for layer in resnet.layers:
          layer.trainable = False
  else :              # For layers training
      print('Resnet layers are trainable...')
      for layer in resnet.layers :
          layer.trainable=True
    
    

  c3 = resnet.get_layer('conv3_block4_out').output
  c4 = resnet.get_layer('conv4_block6_out').output
  c5 = resnet.get_layer('conv5_block3_out').output
  #modified_c5 = resnet_50_last_stage_custom(c4)
  print('Resnet50 is loaded with Imagenet weights...')
  print('c1 , c2 , c3 layers loaded sucessfully..')
  
  return c3 , c4 , c5

def resnet_50_last_stage_custom(last_stage_in):
  #modified Resnet50 last stage
  #Block 1
  block_1 = Conv2D(filters = 512,strides = (2,2),kernel_size = (1,1))(last_stage_in)
  block_1 = layers.BatchNormalization()(block_1)
  block_1 = tf.nn.relu(block_1)
  block_1 = Conv2D(filters = 512,kernel_size = (3,3),padding='same')(block_1)
  block_1 = layers.BatchNormalization()(block_1)
  block_1 = tf.nn.relu(block_1)
  block_1 = Conv2D(filters = 2048,kernel_size = (1,1))(block_1)  
  block_1 = layers.BatchNormalization()(block_1)
  block_1 = tf.nn.relu(block_1)

  #adding_1
  last_stage_input = Conv2D(filters=2048 , kernel_size=(1,1),strides=(2,2))(last_stage_in)
  last_stage_input = layers.BatchNormalization()(last_stage_input)
  last_stage_input = tf.nn.relu(last_stage_input)
  block_1 = block_1 + last_stage_input

  #Block 2
  block_2 = Conv2D(filters = 512,kernel_size = (1,1))(block_1)
  block_2 = layers.BatchNormalization()(block_2)
  block_2 = tf.nn.relu(block_2)
  block_2 = Conv2D(filters = 512,kernel_size = (3,3),padding='same')(block_2)
  block_2 = layers.BatchNormalization()(block_2)
  block_2 = tf.nn.relu(block_2)
  block_2 = Conv2D(filters = 2048,kernel_size = (1,1))(block_2)  
  block_2 = layers.BatchNormalization()(block_2)
  block_2 = tf.nn.relu(block_2)

  # adding 2
  block_2 = block_2 + block_1

  #Block 3
  block_3 = Conv2D(filters = 512,kernel_size = (1,1))(block_2)
  block_3 = layers.BatchNormalization()(block_3)
  block_3 = tf.nn.relu(block_3)
  block_3 = Conv2D(filters = 512,kernel_size = (3,3),padding='same')(block_3)
  block_3 = layers.BatchNormalization()(block_3)
  block_3 = tf.nn.relu(block_3)
  block_3 = Conv2D(filters = 2048,kernel_size = (1,1),name= 'conv5_block3_out')(block_3)  
  block_3 = layers.BatchNormalization()(block_3)
  block_3 = tf.nn.relu(block_3)

  # adding 2
  block_3 = block_3 + block_2

  return block_3















 