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
import time 


def main():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config()
    input_size = args.size
    vid_path = args.video

    saved_model_loaded = tf.saved_model.load(args.model, tags=[tag_constants.SERVING])


    vid = cv2.VideoCapture(vid_path)
    # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    # by default allow all classes in .names file
    allowed_classes = list(class_names.values())
    if args.output:
       # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(args.output, codec, fps, (width, height))
    
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
    
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

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
            score_threshold=args.score )

        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        image = utils.draw_bbox(frame, pred_bbox , allowed_classes = allowed_classes)
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(image)
        
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if args.show == 'True':
          print(args.show)
          cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
          cv2.imshow('reuslt',result)

        if args.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv.destroyAllWindows()    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model',help = 'path to model',type = str , default='./checkpoints/saved_model')
    parser.add_argument('-s','--size',help = 'image size',type = int ,default = 416 )
    parser.add_argument('-i','--video',help = 'path of input image', type = str )
    parser.add_argument('-u','--iou',help = 'iou threshold',type = float , default = 0.4)
    parser.add_argument('-c','--score',help = 'score threshold',type = float , default = 0.2 )
    parser.add_argument('-o','--output',help ='path to save output', type = str , default = './output/output.avi')
    parser.add_argument('-w','--show',help ='Show result in window' , type = str, default= 'False' )
    args = parser.parse_args()
    main()










