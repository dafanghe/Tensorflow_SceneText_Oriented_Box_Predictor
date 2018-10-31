# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Detection model trainer.

This file provides a generic training method that can be used to train a
DetectionModel.
"""

import functools

import tensorflow as tf

from builders import optimizer_builder
from builders import preprocessor_builder
from core import batcher
from core import preprocessor
from core import standard_fields as fields
from utils import ops as util_ops
from utils import variables_helper
from deployment import model_deploy
import PIL
from PIL import Image
import cv2
import numpy as np

slim = tf.contrib.slim

LOSS_NAME_MAPPING = {}


def visualize_tensor_dict(tensor_dict):
  colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
      (255, 0, 255), (255, 125, 0), (0, 125, 255), (0, 255, 125), (125, 255, 0),
      (255, 0, 125), (125, 0, 255)]
  image = tensor_dict['image'][0]
  if image.dtype == np.float32:
    image = (image - np.min(image))/(np.max(image) - np.min(image)) * 255
    image = image.astype(np.uint8)
  image2 = np.copy(image)
  boxes = tensor_dict['groundtruth_boxes']
  num_box = boxes.shape[0]
  for i in range(num_box):
    box = boxes[i]
    ymin = int(boxes[i][0] * image.shape[0])
    ymax = int(boxes[i][2] * image.shape[0])
    xmin = int(boxes[i][1] * image.shape[1])
    xmax = int(boxes[i][3] * image.shape[1])
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax),
        colors[i%len(colors)], 3)
  outname = 'test/test.jpg'
  pil_im = Image.fromarray(image)
  pil_im.save(outname)
  oriented_boxes = tensor_dict['oriented_boxes']
  for i in range(0, oriented_boxes.shape[0]):
      box = oriented_boxes[i]
      for j in range(0, 4):
          point = (int(box[2*j] * image2.shape[1]), int(box[2*j+1] * image2.shape[0]))
          cv2.circle(image2, point, 3, colors[j], -1)
  outname = 'test/test2.jpg'
  pil_im = Image.fromarray(image2)
  pil_im.save(outname)
  import pdb
  pdb.set_trace()


def _create_input_queue(batch_size_per_clone, create_tensor_dict_fn,
                        batch_queue_capacity, num_batch_queue_threads,
                        prefetch_queue_capacity, data_augmentation_options):
  """Sets up reader, prefetcher and returns input queue.

  Args:
    batch_size_per_clone: batch size to use per clone.
    create_tensor_dict_fn: function to create tensor dictionary.
    batch_queue_capacity: maximum number of elements to store within a queue.
    num_batch_queue_threads: number of threads to use for batching.
    prefetch_queue_capacity: maximum capacity of the queue used to prefetch
                             assembled batches.
    data_augmentation_options: a list of tuples, where each tuple contains a
      data augmentation function and a dictionary containing arguments and their
      values (see preprocessor.py).

  Returns:
    input queue: a batcher.BatchQueue object holding enqueued tensor_dicts
      (which hold images, boxes and targets).  To get a batch of tensor_dicts,
      call input_queue.Dequeue().
  """
  tensor_dict = create_tensor_dict_fn()

  tensor_dict[fields.InputDataFields.image] = tf.expand_dims(
      tensor_dict[fields.InputDataFields.image], 0)

  images = tensor_dict[fields.InputDataFields.image]
  float_images = tf.to_float(images)
  tensor_dict[fields.InputDataFields.image] = float_images

  tensor_dict[fields.InputDataFields.groundtruth_oriented_boxes] = tf.reshape(
          tensor_dict[fields.InputDataFields.groundtruth_oriented_boxes], [-1, 4, 2])
  if data_augmentation_options:
    tensor_dict = preprocessor.preprocess(tensor_dict,
                                          data_augmentation_options)
  input_queue = batcher.BatchQueue(
      tensor_dict,
      batch_size=batch_size_per_clone,
      batch_queue_capacity=batch_queue_capacity,
      num_batch_queue_threads=num_batch_queue_threads,
      prefetch_queue_capacity=prefetch_queue_capacity)
  if False:
    init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
    with tf.Session() as sess:
      sess.run(init_op)
      tf.train.start_queue_runners(sess)
      result = sess.run(input_queue)
      import pdb
      pdb.set_trace()
    import pdb
    pdb.set_trace()
  return input_queue


def _get_inputs(input_queue, num_classes):
  """Dequeue batch and construct inputs to object detection model.

  Args:
    input_queue: BatchQueue object holding enqueued tensor_dicts.
    num_classes: Number of classes.

  Returns:
    images: a list of 3-D float tensor of images.
    locations_list: a list of tensors of shape [num_boxes, 4]
      containing the corners of the groundtruth boxes.
    classes_list: a list of padded one-hot tensors containing target classes.
    masks_list: a list of 3-D float tensors of shape [num_boxes, image_height,
      image_width] containing instance masks for objects if present in the
      input_queue. Else returns None.
  """
  read_data_list = input_queue.dequeue()
  label_id_offset = 1
  def extract_images_and_targets(read_data):
    image = read_data[fields.InputDataFields.image]
    location_gt = read_data[fields.InputDataFields.groundtruth_boxes]
    location_gt_oriented = read_data[fields.InputDataFields.groundtruth_oriented_boxes]
    classes_gt = tf.cast(read_data[fields.InputDataFields.groundtruth_classes],
                         tf.int32)
    classes_gt -= label_id_offset
    classes_gt = util_ops.padded_one_hot_encoding(indices=classes_gt,
                                                  depth=num_classes, left_pad=0)
    masks_gt = read_data.get(fields.InputDataFields.groundtruth_instance_masks)
    return image, location_gt, location_gt_oriented, classes_gt, masks_gt
  return zip(*map(extract_images_and_targets, read_data_list))


def _create_losses(input_queue, create_model_fn):
  """Creates loss function for a DetectionModel.

  Args:
    input_queue: BatchQueue object holding enqueued tensor_dicts.
    create_model_fn: A function to create the DetectionModel.
  """
  detection_model = create_model_fn()
  (images, groundtruth_boxes_list, groundtruth_oriented_boxes_list, groundtruth_classes_list,
   groundtruth_masks_list) = _get_inputs(input_queue, detection_model.num_classes)

  if False:
    init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
    config = tf.ConfigProto(
                allow_soft_placement=True)
    with tf.Session(config=config) as sess:
      sess.run(init_op)
      tf.train.start_queue_runners(sess)
      while True:
        results = sess.run([images,
            groundtruth_boxes_list,
            groundtruth_oriented_boxes_list,
            groundtruth_classes_list])
        im = results[0][0][0].astype(np.uint8)
        boxes = results[1][0]
        oriented_boxes = results[2][0]

        draw_image = np.copy(im)
        for i in range(boxes.shape[0]):
          miny = int(boxes[i][0] * im.shape[0])
          minx = int(boxes[i][1] * im.shape[1])
          maxy = int(boxes[i][2] * im.shape[0])
          maxx = int(boxes[i][3] * im.shape[1])
          cv2.rectangle(draw_image, (minx, miny), (maxx, maxy), (0, 0, 255), 2)
        outname = 'test1.jpg'
        cv2.imwrite(outname, draw_image)

        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
        for i in range(oriented_boxes.shape[0]):
          for j in range(4):
            point1 = (int(oriented_boxes[i][j][1] * im.shape[1]),
                    int(oriented_boxes[i][j][0] * im.shape[0]))
            cv2.circle(im, point1, 3, colors[j], -1)
            #point2 = (int(oriented_boxes[i][(j+1)%4][1] * im.shape[1]),
            #        int(oriented_boxes[i][(j+1)%4][0] * im.shape[0]))
            #cv2.line(im, point1, point2, colors[j], 2)
        outname = 'test2.jpg'
        cv2.imwrite(outname, im)
        import pdb
        pdb.set_trace()
      print('a')

  images = [detection_model.preprocess(image) for image in images]
  images = tf.concat(images, 0)
  if any(mask is None for mask in groundtruth_masks_list):
    groundtruth_masks_list = None

  detection_model.provide_groundtruth(groundtruth_boxes_list,
                                      groundtruth_oriented_boxes_list,
                                      groundtruth_classes_list,
                                      groundtruth_masks_list)
  prediction_dict = detection_model.predict(images)
  losses_dict = detection_model.loss(prediction_dict)
  for loss_name, loss_tensor in losses_dict.iteritems():
    tf.losses.add_loss(loss_tensor)
    LOSS_NAME_MAPPING[loss_tensor.op.name] = loss_name

def train(create_tensor_dict_fn, create_model_fn, train_config, master, task,
          num_clones, worker_replicas, clone_on_cpu, ps_tasks, worker_job_name,
          is_chief, train_dir):
  """Training function for detection models.

  Args:
    create_tensor_dict_fn: a function to create a tensor input dictionary.
    create_model_fn: a function that creates a DetectionModel and generates
                     losses.
    train_config: a train_pb2.TrainConfig protobuf.
    master: BNS name of the TensorFlow master to use.
    task: The task id of this training instance.
    num_clones: The number of clones to run per machine.
    worker_replicas: The number of work replicas to train with.
    clone_on_cpu: True if clones should be forced to run on CPU.
    ps_tasks: Number of parameter server tasks.
    worker_job_name: Name of the worker job.
    is_chief: Whether this replica is the chief replica.
    train_dir: Directory to write checkpoints and training summaries to.
  """

  detection_model = create_model_fn()
  data_augmentation_options = [
      preprocessor_builder.build(step)
      for step in train_config.data_augmentation_options]
  with tf.Graph().as_default():
    # Build a configuration specifying multi-GPU and multi-replicas.
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=num_clones,
        clone_on_cpu=clone_on_cpu,
        replica_id=task,
        num_replicas=worker_replicas,
        num_ps_tasks=ps_tasks,
        worker_job_name=worker_job_name)

    # Place the global step on the device storing the variables.
    with tf.device(deploy_config.variables_device()):
      global_step = slim.create_global_step()

    with tf.device(deploy_config.inputs_device()):
      input_queue = _create_input_queue(train_config.batch_size // num_clones,
                                        create_tensor_dict_fn,
                                        train_config.batch_queue_capacity,
                                        train_config.num_batch_queue_threads,
                                        train_config.prefetch_queue_capacity,
                                        data_augmentation_options)

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    global_summaries = set([])

    model_fn = functools.partial(_create_losses,
                                 create_model_fn=create_model_fn)
    clones = model_deploy.create_clones(deploy_config, model_fn, [input_queue])
    first_clone_scope = clones[0].scope

    # Gather update_ops from the first clone. These contain, for example,
    # the updates for the batch_norm variables created by model_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    with tf.device(deploy_config.optimizer_device()):
      training_optimizer = optimizer_builder.build(train_config.optimizer,
                                                   global_summaries)

    sync_optimizer = None
    if train_config.sync_replicas:
      training_optimizer = tf.SyncReplicasOptimizer(
          training_optimizer,
          replicas_to_aggregate=train_config.replicas_to_aggregate,
          total_num_replicas=train_config.worker_replicas)
      sync_optimizer = training_optimizer

    # Create ops required to initialize the model from a given checkpoint.
    init_fn = None
    if train_config.fine_tune_checkpoint:
      var_map = detection_model.restore_map(
          from_detection_checkpoint=train_config.from_detection_checkpoint)
      available_var_map = (variables_helper.
                           get_variables_available_in_checkpoint(
                               var_map, train_config.fine_tune_checkpoint))
      init_saver = tf.train.Saver(available_var_map)
      def initializer_fn(sess):
        init_saver.restore(sess, train_config.fine_tune_checkpoint)
      init_fn = initializer_fn

    with tf.device(deploy_config.optimizer_device()):
      total_loss, grads_and_vars = model_deploy.optimize_clones(
          clones, training_optimizer, regularization_losses=None)
      total_loss = tf.check_numerics(total_loss, 'LossTensor is inf or nan.')

      # Optionally multiply bias gradients by train_config.bias_grad_multiplier.
      if train_config.bias_grad_multiplier:
        biases_regex_list = ['.*/biases']
        grads_and_vars = variables_helper.multiply_gradients_matching_regex(
            grads_and_vars,
            biases_regex_list,
            multiplier=train_config.bias_grad_multiplier)
      # Optionally freeze some layers by setting their gradients to be zero.
      if train_config.freeze_variables:
        grads_and_vars = variables_helper.freeze_gradients_matching_regex(
            grads_and_vars, train_config.freeze_variables)

      # Optionally clip gradients
      if train_config.gradient_clipping_by_norm > 0:
        with tf.name_scope('clip_grads'):
          grads_and_vars = slim.learning.clip_gradient_norms(
              grads_and_vars, train_config.gradient_clipping_by_norm)

      # Create gradient updates.
      grad_updates = training_optimizer.apply_gradients(grads_and_vars,
                                                        global_step=global_step)
      update_ops.append(grad_updates)

      update_op = tf.group(*update_ops)
      with tf.control_dependencies([update_op]):
        train_tensor = tf.identity(total_loss, name='train_op')

    # Add summaries.
    #for model_var in slim.get_model_variables():
    #  global_summaries.add(tf.summary.histogram(model_var.op.name, model_var))
    for loss_tensor in tf.losses.get_losses():
      summary_name = LOSS_NAME_MAPPING[loss_tensor.op.name]
      global_summaries.add(tf.summary.scalar(summary_name, loss_tensor))

    global_summaries.add(
        tf.summary.scalar('TotalLoss', tf.losses.get_total_loss()))

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))
    summaries |= global_summaries

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    # Soft placement allows placing on CPU ops without GPU implementation.
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False,
                                    gpu_options=gpu_options)

    # Save checkpoints regularly.
    keep_checkpoint_every_n_hours = train_config.keep_checkpoint_every_n_hours
    saver = tf.train.Saver(
        keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

    slim.learning.train(
        train_tensor,
        logdir=train_dir,
        master=master,
        is_chief=is_chief,
        session_config=session_config,
        startup_delay_steps=train_config.startup_delay_steps,
        init_fn=init_fn,
        summary_op=summary_op,
        number_of_steps=(
            train_config.num_steps if train_config.num_steps else None),
        save_summaries_secs=120,
        sync_optimizer=sync_optimizer,
        saver=saver)
