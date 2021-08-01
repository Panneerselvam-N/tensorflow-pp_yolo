import os
import shutil
import tensorflow as tf
from utils.data_process import decode_tf , filter_boxes
from utils.config import cfg
import numpy as np
from utils import utils
from core.fpn import fpn
from core.resnet_50 import resnet_50
from core.head import head
import argparse

def save_tf():
  STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config()

  resnet = tf.keras.applications.ResNet50(include_top=False,weights='imagenet',input_shape=(args.size,args.size,3))
  c3 , c4 ,c5 = resnet_50(resnet) 
  neck_output = fpn(c3 , c4 , c5)
  head_output = head(neck_output)
  feature_maps = head_output
  bbox_tensors = []
  prob_tensors = []

  input_size = 416
  for i, fm in enumerate(feature_maps):
    if i == 0:
      output_tensors = decode_tf(fm, args.size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
    elif i == 1:
      output_tensors = decode_tf(fm, args.size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE )
    else:
      output_tensors = decode_tf(fm, args.size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
    bbox_tensors.append(output_tensors[0])
    prob_tensors.append(output_tensors[1])
  pred_bbox = tf.concat(bbox_tensors, axis=1)
  pred_prob = tf.concat(prob_tensors, axis=1)

  boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=0.2, input_shape=tf.constant([args.size, args.size]))
  pred = tf.concat([boxes, pred_conf], axis=-1)
  model = tf.keras.Model(resnet.input, pred)
  print('loading weights..')
  model.load_weights(args.weights)
  #model.summary()
  print('saving model..')
  model.save(args.save)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-s','--size',help = 'Size of input image',type = int ,default = 416)
  parser.add_argument('-w','--weights',help = 'path to weights',type = str , default = './checkpoints/pp_yolo')
  parser.add_argument('-m','--save', help = 'path to save model',type = str , default = './checkpoints/saved_model')

  args = parser.parse_args()
  save_tf()