import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
from utils import utils
from utils.config import cfg
import argparse
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def main():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config()
    input_size = args.size
    image_path = args.image

    
    
    saved_model_loaded = tf.saved_model.load(args.model, tags=[tag_constants.SERVING])

   
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.
    
    image_data = np.asarray(image_data).astype(np.float32)
    image_data = np.expand_dims(image_data,axis=0)
       
        
    infer = saved_model_loaded.signatures['serving_default']
    batch_data = tf.constant(image_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=args.iou,
            score_threshold=args.score
        )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

    # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    # by default allow all classes in .names file
    allowed_classes = list(class_names.values())
        
        

    image = utils.draw_bbox(original_image, pred_bbox, allowed_classes = allowed_classes)

    image = Image.fromarray(image.astype(np.uint8))

    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite(args.output, image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model',help = 'path to model',type = str , default='./checkpoints/saved_model')
    parser.add_argument('-s','--size',help = 'image size',type = int ,default = 416 )
    parser.add_argument('-i','--image',help = 'path of input image', type = str )
    parser.add_argument('-u','--iou',help = 'iou threshold',type = float , default = 0.4)
    parser.add_argument('-c','--score',help = 'score threshold',type = float , default = 0.2 )
    parser.add_argument('-o','--output',help ='path to save output', type = str , default = './output/output.jpg')

    args = parser.parse_args()
    main()