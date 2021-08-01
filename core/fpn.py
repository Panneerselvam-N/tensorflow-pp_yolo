from numpy import short
import tensorflow as tf 
from tensorflow.keras.layers import Conv2D
from core.blocks import coordconv , convblock ,sppblock , upsampling

def fpn(c3,c4,c5):
    conv_1 = coordconv(c5)
    conv_1 = Conv2D(filters=512 , kernel_size=(1,1),padding='same')(conv_1)
    conv_1 = tf.nn.relu(conv_1)
    conv_1 = convblock(conv_1)
    conv_1 = sppblock(conv_1)
    conv_1 = convblock(conv_1)
    shortcut_1 = conv_1
    
    conv_2 = upsampling(shortcut_1)
    conv_2 = tf.concat([c4 , conv_2],axis =-1)
    conv_2 = coordconv(conv_2)
    conv_2 = Conv2D(filters=256, kernel_size=(1,1),padding='same')(conv_2)
    conv_2 = tf.nn.relu(conv_2) 
    conv_2 = convblock(conv_2)
    conv_2 = convblock(conv_2) 
    shortcut_2 = conv_2 
    
    conv_3 = upsampling(shortcut_2)
    conv_3 = tf.concat([c3,conv_3],axis = -1)
    conv_3 =coordconv(conv_3)
    conv_3 = Conv2D(filters=128 , kernel_size=(1,1))(conv_3)
    conv_3 = tf.nn.relu(conv_3)
    conv_3 = convblock(conv_3)
    conv_3 = convblock(conv_3)

    return [conv_1,conv_2,conv_3]