"""Functions to export object detection inference graph."""
from protos import pipeline_pb2
from google.protobuf import text_format
import logging
import os
import collections
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import saver as saver_lib
from builders import model_builder
from core import standard_fields as fields
from data_decoders import tf_example_decoder
import cv2
import numpy as np
import math

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('pipeline_config_path', None,
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file.')

flags.DEFINE_string('train_dir', None,
                    'path/to/')

flags.DEFINE_string('image_dir', None,
        '')

flags.DEFINE_string('output_dir', None,
        '')

FLAGS = flags.FLAGS


colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (125, 125, 0), (0, 125, 125), (125, 0, 125)]

def construct_graph(pipeline_config, inputs):
  print('construct graph')
  detection_model = model_builder.build(pipeline_config.model,
                                        is_training=False)
  inputs = tf.to_float(inputs)
  preprocessed_inputs = detection_model.preprocess(inputs)
  output_tensors = detection_model.predict(preprocessed_inputs)
  postprocessed_tensors = detection_model.postprocess(output_tensors)
  return output_tensors, postprocessed_tensors


def visualize_horizontal_box(img, boxes, scales):
  scale_h, scale_w = scales
  boxes[:, 0] *= scale_h
  boxes[:, 1] *= scale_w
  boxes[:, 2] *= scale_h
  boxes[:, 3] *= scale_w
  boxes = boxes.astype(np.int32)
  for i in range(boxes.shape[0]):
    box = boxes[i]
    cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), colors[i%len(colors)], 2)


def main(_):
  assert FLAGS.pipeline_config_path, '`pipeline_config_path` is missing'
  assert FLAGS.train_dir, 'train_dir is missing'
  assert FLAGS.output_dir, 'output_dir is missing'
  assert FLAGS.image_dir, 'image_dir is missing'
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  tf.gfile.MakeDirs(FLAGS.output_dir)
  with tf.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)

  latest_checkpoint = tf.train.latest_checkpoint(FLAGS.train_dir)
  with tf.get_default_graph().as_default():
    input_tensor = tf.placeholder(dtype=tf.uint8,
                                shape=(1, None, None, 3),
                                name='image_tensor')
    output_tensors, postprocessed_tensors = construct_graph(pipeline_config, input_tensor)
    saver = tf.train.Saver()
    with tf.Session() as sess:
      saver.restore(sess, latest_checkpoint)
      files = os.listdir(FLAGS.image_dir)
      for f in files:
        if not f.endswith('.jpg'):
          continue
        print('do inference on:', f)
        im = cv2.imread(FLAGS.image_dir + f)
        results = sess.run([output_tensors, postprocessed_tensors],
                feed_dict={input_tensor:[im]})
        scale_h = im.shape[0]/float(results[0]['image_shape'][1])
        scale_w = im.shape[1]/float(results[0]['image_shape'][2])
        scores = results[1]['detection_scores'][0]

        #visualize proposal box
#        proposal_img = np.copy(im)
#        proposal_boxes = results[0]['proposal_boxes'][0]
#        visualize_horizontal_box(proposal_img, proposal_boxes, (scale_h, scale_w))
#        outname = FLAGS.output_dir + im_name + '_proposals.jpg'
#        cv2.imwrite(outname, proposal_img)

        oriented_boxes = results[1]['detection_oriented_boxes'][0]
        oriented_boxes[:, :, 0] *= im.shape[0]
        oriented_boxes[:, :, 1] *= im.shape[1]
        oriented_box_img = np.copy(im)
        for i in range(0, scores.shape[0]):
          cur_score = scores[i]
          cur_ibox = oriented_boxes[i]
          if cur_score > 0.6:
            cur_ibox = cur_ibox.astype(np.int32)
            for j in range(0, 4):
              cur_color = colors[j%len(colors)]
              point1 = (cur_ibox[j][1], cur_ibox[j][0])
              point2 = (cur_ibox[(j+1)%4][1], cur_ibox[(j+1)%4][0])
              cv2.line(oriented_box_img, point1, point2, cur_color, 2)
        outname = FLAGS.output_dir + f
        cv2.imwrite(outname, oriented_box_img)

        #visualize final detected boxes
#        boxes = results[1]['detection_boxes'][0]
#        for i in range(0, scores.shape[0]):
#          cur_score = scores[i]
#          cur_box = boxes[i]
#          cur_color = colors[i%len(colors)]
#          if cur_score > 0.6:
#            ymin = int(cur_box[0] * im.shape[0])
#            xmin = int(cur_box[1] * im.shape[1])
#            ymax = int(cur_box[2] * im.shape[0])
#            xmax = int(cur_box[3] * im.shape[1])
#            cv2.rectangle(im, (xmin, ymin), (xmax, ymax), cur_color, 2)
#        outname = FLAGS.output_dir + im_name + '_detected_box.jpg'
#        cv2.imwrite(outname, im)

        #write the results
        count = 0
        outname = FLAGS.output_dir + 'res_' + f[:-4] + '.txt'
        fout = open(outname, 'w')
        for i in range(0, scores.shape[0]):
          cur_score = scores[i]
          cur_ibox = oriented_boxes[i]
          cur_color = colors[i%len(colors)]
          if cur_score > 0.6:
            count += 1
            cur_ibox = cur_ibox.astype(np.int32)
            for j in range(0, 4):
              fout.write(str(cur_ibox[j][1]) + ',' + str(cur_ibox[j][0]))
              if j!= 3:
                fout.write(',')
            fout.write('\n')
        fout.close()
        print 'detected ', count, ' boxes'

if __name__ == '__main__':
  tf.app.run()
