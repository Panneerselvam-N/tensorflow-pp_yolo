import os
import shutil
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
from utils.data_process import compute_loss, decode_train
from utils.dataset import Dataset
from utils.config import cfg
import numpy as np
from utils import utils
from utils.utils import freeze_all, unfreeze_all
from core.fpn import fpn
from core.resnet_50 import resnet_50
from core.head import head
import argparse

def main():
    print('loading datasets...')
    trainset = Dataset(is_training=True)
    #testset = Dataset(is_training=False)
    logdir = "./data/log"
    isfreeze = False
    steps_per_epoch = len(trainset)
    first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
    second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
    total_steps = (first_stage_epochs + second_stage_epochs) * steps_per_epoch
    # train_steps = (first_stage_epochs + second_stage_epochs) * steps_per_period

    #input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config()
    IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH

    freeze_layers = utils.load_freeze_layer()
    resnet = tf.keras.applications.ResNet50(include_top=False,weights='imagenet',input_shape=(cfg.TRAIN.INPUT_SIZE,cfg.TRAIN.INPUT_SIZE,3))
    c3 , c4 ,c5 = resnet_50(resnet) 
    neck_output = fpn(c3 , c4 , c5)
    head_output = head(neck_output)
    feature_maps = head_output
    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
        elif i == 1:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
        else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
        bbox_tensors.append(fm)
        bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(resnet.input, bbox_tensors)
    #model.summary()
    if args.resume != None:
      print('Resuming with last weights -> .{}'.format(args.resume))
      model.load_weights(args.resume)
      
    else : 
      print("Training from scratch")
       


    optimizer = tf.keras.optimizers.Adam(learning_rate= 0.00001)
    if os.path.exists(logdir): shutil.rmtree(logdir)
    writer = tf.summary.create_file_writer(logdir)

    # define training step function
    # @tf.function
    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            tf.print("=> STEP %4d/%4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, total_steps, optimizer.lr.numpy(),
                                                               giou_loss, conf_loss,
                                                               prob_loss, total_loss))
            # update learning rate
            global_steps.assign_add(1)
            if args.const_lr == True: 
              optimizer.lr.assign(args.lr)
            else:
              if global_steps < warmup_steps:
                  lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
              else:
                  lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                      (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                  )
              optimizer.lr.assign(lr.numpy())
            

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()
    def test_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            tf.print("=> TEST STEP %4d   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, giou_loss, conf_loss,
                                                               prob_loss, total_loss))

    for epoch in range(first_stage_epochs + second_stage_epochs):
        if epoch < first_stage_epochs:
            if not isfreeze:
                isfreeze = True
                for name in freeze_layers:
                    freeze = model.get_layer(name)
                    freeze_all(freeze)
        elif epoch >= first_stage_epochs:
            if isfreeze:
                isfreeze = False
                for name in freeze_layers:
                    freeze = model.get_layer(name)
                    unfreeze_all(freeze)
        for image_data, target in trainset:
            train_step(image_data, target)
        #for image_data, target in testset:
          #  test_step(image_data, target)
        print('saving checkpoints...-> {}'.format(args.checkpoints))
        model.save_weights(args.checkpoints)
        
        

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-r','--resume',help = 'path to last weights',type = str)
  parser.add_argument('-c','--checkpoints',help = 'path to save checkpoints',type = str ,default = './checkpoints/pp_yolo')
  parser.add_argument('-l','--lr',help = 'learning rate for resuming',type= float , default= cfg.CONST_LR)
  parser.add_argument('-n','--const_lr',help = 'set constant lr',type = bool,default = False)
  args = parser.parse_args()
  main()
