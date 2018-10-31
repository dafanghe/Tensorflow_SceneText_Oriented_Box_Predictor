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

"""Faster RCNN box coder.

Faster RCNN box coder follows the coding schema described below:
  ty = (y - ya) / ha
  tx = (x - xa) / wa
  th = log(h / ha)
  tw = log(w / wa)
  where x, y, w, h denote the box's center coordinates, width and height

  respectively. Similarly, xa, ya, wa, ha denote the anchor's center
  coordinates, width and height. tx, ty, tw and th denote the anchor-encoded
  center, width and height respectively.

  See http://arxiv.org/abs/1506.01497 for details.
"""

import tensorflow as tf

from core import box_coder
from core import box_list

EPSILON = 1e-8


class FasterRcnnBoxCoder(box_coder.BoxCoder):
  """Faster RCNN box coder."""

  def __init__(self, scale_factors=None, scale_factors_oriented=None):
    """Constructor for FasterRcnnBoxCoder.

    Args:
      scale_factors: List of 4 positive scalars to scale ty, tx, th and tw.
        If set to None, does not perform scaling. For Faster RCNN,
        the open-source implementation recommends using [10.0, 10.0, 5.0, 5.0].
    """
    if scale_factors:
      assert len(scale_factors) == 4
      for scalar in scale_factors:
        assert scalar > 0
    if scale_factors_oriented:
      assert len(scale_factors_oriented) == 8
      for scalar in scale_factors:
        assert scalar > 0
    self._scale_factors_oriented = scale_factors_oriented
    self._scale_factors = scale_factors

  @property
  def code_size(self):
    return 4

  @property
  def code_size_oriented(self):
    return 8

  def _encode(self, boxes, anchors):
    """Encode a box collection with respect to anchor collection.

    Args:
      boxes: BoxList holding N boxes to be encoded.
      anchors: BoxList of anchors.

    Returns:
      a tensor representing N anchor-encoded boxes of the format
      [ty, tx, th, tw].
    """
    # Convert anchors to the center coordinate representation.
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
    ycenter, xcenter, h, w = boxes.get_center_coordinates_and_sizes()
    # Avoid NaN in division and log below.
    ha += EPSILON
    wa += EPSILON
    h += EPSILON
    w += EPSILON

    tx = (xcenter - xcenter_a) / wa
    ty = (ycenter - ycenter_a) / ha
    tw = tf.log(w / wa)
    th = tf.log(h / ha)
    # Scales location targets as used in paper for joint training.
    if self._scale_factors:
      ty *= self._scale_factors[0]
      tx *= self._scale_factors[1]
      th *= self._scale_factors[2]
      tw *= self._scale_factors[3]
    return tf.transpose(tf.stack([ty, tx, th, tw]))

  def _encode_oriented(self, boxes, anchors):
    """Encode a box collection with respect to anchor collection.

    Args:
      boxes: BoxList holding N boxes to be encoded.
      anchors: BoxList of anchors.

    Returns:
      a tensor representing N anchor-encoded boxes of the format
      [y1, x1, y2, x2, y3, x3, y4, x4].
    """
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
    # Convert anchors to the center coordinate representation.
    oriented_boxes = boxes.get_oriented()
    oriented_boxes = tf.reshape(oriented_boxes, [-1, 8])
    y1, x1, y2, x2, y3, x3, y4, x4 = tf.unstack(tf.transpose(oriented_boxes))

    # Avoid NaN in division and log below.
    ha += EPSILON
    wa += EPSILON

    tx1 = (x1 - xcenter_a) / wa
    tx2 = (x2 - xcenter_a) / wa
    tx3 = (x3 - xcenter_a) / wa
    tx4 = (x4 - xcenter_a) / wa

    ty1 = (y1 - ycenter_a) / ha
    ty2 = (y2 - ycenter_a) / ha
    ty3 = (y3 - ycenter_a) / ha
    ty4 = (y4 - ycenter_a) / ha

    # Scales location targets as used in paper for joint training.
    #hard coded for now
    if self._scale_factors_oriented:
      ty1 *= self._scale_factors_oriented[0]
      tx1 *= self._scale_factors_oriented[1]
      ty2 *= self._scale_factors_oriented[2]
      tx2 *= self._scale_factors_oriented[3]
      ty3 *= self._scale_factors_oriented[4]
      tx3 *= self._scale_factors_oriented[5]
      ty4 *= self._scale_factors_oriented[6]
      tx4 *= self._scale_factors_oriented[7]
    return tf.transpose(tf.stack([ty1, tx1, ty2, tx2, ty3, tx3, ty4, tx4]))

  def _decode(self, rel_codes, anchors):
    """Decode relative codes to boxes.

    Args:
      rel_codes: a tensor representing N anchor-encoded boxes.
      anchors: BoxList of anchors.

    Returns:
      boxes: BoxList holding N bounding boxes.
    """
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()

    ty, tx, th, tw = tf.unstack(tf.transpose(rel_codes))
    if self._scale_factors:
      ty /= self._scale_factors[0]
      tx /= self._scale_factors[1]
      th /= self._scale_factors[2]
      tw /= self._scale_factors[3]
    w = tf.exp(tw) * wa
    h = tf.exp(th) * ha
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    ymin = ycenter - h / 2.
    xmin = xcenter - w / 2.
    ymax = ycenter + h / 2.
    xmax = xcenter + w / 2.
    return box_list.BoxList(tf.transpose(tf.stack([ymin, xmin, ymax, xmax])))


  def _decode_oriented(self, rel_codes, anchors):
    """Decode relative codes to boxes.

    Args:
      rel_codes: a tensor representing N anchor-encoded oriented boxes.
      anchors: BoxList of anchors.

    Returns:
      boxes: BoxList holding N bounding boxes.
    """

    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()

    ty1, tx1, ty2, tx2, ty3, tx3, ty4, tx4 = tf.unstack(tf.transpose(rel_codes))
    if self._scale_factors_oriented:
      ty1 /= self._scale_factors_oriented[0]
      tx1 /= self._scale_factors_oriented[1]
      ty2 /= self._scale_factors_oriented[2]
      tx2 /= self._scale_factors_oriented[3]
      ty3 /= self._scale_factors_oriented[4]
      tx3 /= self._scale_factors_oriented[5]
      ty4 /= self._scale_factors_oriented[6]
      tx4 /= self._scale_factors_oriented[7]

    y1 = ty1 * ha + ycenter_a
    x1 = tx1 * wa + xcenter_a
    y2 = ty2 * ha + ycenter_a
    x2 = tx2 * wa + xcenter_a
    y3 = ty3 * ha + ycenter_a
    x3 = tx3 * wa + xcenter_a
    y4 = ty4 * ha + ycenter_a
    x4 = tx4 * wa + xcenter_a
    oriented_boxes = tf.transpose(tf.stack([y1, x1, y2, x2, y3, x3, y4, x4]))
    oriented_boxes = tf.reshape(oriented_boxes, [-1, 4, 2])
    return box_list.BoxList(oriented_boxes=oriented_boxes)
