import tensorflow as tf
from tensorflow.keras.layers import Conv2D , BatchNormalization , MaxPool2D
from utils.config import cfg 

def convblock(input_tensors, bn=True , coorconv = True ):
    
    conv = Conv2D(filters= input_tensors.shape[-1]*2, kernel_size=(3,3),padding='same')(input_tensors)

    if bn: conv = BatchNormalization()(conv)

    conv = tf.nn.relu(conv)
    if coordconv: conv = coordconv(conv)
    conv = Conv2D(filters=input_tensors.shape[-1], kernel_size =(1,1))(conv)
    if bn: conv = BatchNormalization()(conv)
    conv = tf.nn.relu(conv)

    return conv

def coordconv(feature_map):
    batch_size = tf.shape(feature_map)[0]
    x_shape = tf.shape(feature_map)[1]
    y_shape = tf.shape(feature_map)[2]

    x_ones = tf.ones((batch_size , x_shape),dtype=tf.float32)
    x_ones = tf.expand_dims(x_ones,axis = -1)
    x_range = tf.tile(tf.expand_dims(tf.range(y_shape,dtype=tf.float32),axis=0),[batch_size,1])
    x_range = tf.expand_dims(x_range,1)
    x_channel = tf.matmul(x_ones,x_range)
    x_channel = tf.expand_dims(x_channel,axis=-1)

    y_ones = tf.ones((batch_size , y_shape),dtype=tf.float32)
    y_ones = tf.expand_dims(y_ones,axis = 1)
    y_range = tf.tile(tf.expand_dims(tf.range(x_shape,dtype=tf.float32),axis=0),[batch_size,1])
    y_range = tf.expand_dims(y_range,-1)
    y_channel = tf.matmul(y_range,y_ones)
    y_channel = tf.expand_dims(y_channel,axis=-1)

    x_shape = tf.cast(x_shape , dtype=tf.float32)
    y_shape = tf.cast(y_shape, dtype = tf.float32)


    x_channel = tf.cast(x_channel,dtype=tf.float32) / (y_shape -1)
    y_channel = tf.cast(y_channel,dtype=tf.float32) / (x_shape - 1)

    x_channel = x_channel * 2 - 1
    y_channel = y_channel * 2 -1

    output_tensors = tf.concat([feature_map,x_channel,y_channel],axis=-1)

    return output_tensors

def upsampling(features):
    channels = features.shape[-1]
    conv = coordconv(features)
    conv = Conv2D(filters=channels/2 ,kernel_size=(1,1))(conv)
    output = tf.image.resize(conv,size=(conv.shape[1]*2,conv.shape[2]*2))
    return output

def sppblock(input_tensors):

    pooling_1 =  MaxPool2D(pool_size=(1,1),strides=(1,1))(input_tensors)
    pooling_2 = MaxPool2D(pool_size=(5,5),padding='same',strides=(1,1))(input_tensors)
    pooling_3 = MaxPool2D(pool_size=(9,9),padding='same',strides=(1,1))(input_tensors)
    #pooling_4 = MaxPool2D(pool_size=(13,13),padding='same',strides=(1,1))(input_tensors)

    output = tf.concat([input_tensors,pooling_1,pooling_2 ,pooling_3],axis=-1)

    return output

def conv_head(features):
    channel = features.shape[-1]
    num_classes = cfg.YOLO.NUM_CLASSES 
    num_filters = 3 * (num_classes + 5)
    conv = coordconv(features)
    conv = Conv2D(filters=channel*2,kernel_size=(3,3),padding='same')(conv)
    conv = tf.nn.relu(conv)
    conv = Conv2D(filters= num_filters,kernel_size=(1,1))(conv)

    return conv 

