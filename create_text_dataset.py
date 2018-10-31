import os
import sys
import glob
import numpy as np
import random
import math
import io
import re
from PIL import Image
import PIL
import hashlib
import tensorflow as tf
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from object_detection.utils import dataset_util
from shapely import geometry
import cv2

LABEL_MAP = {
   'background': 0,
   'text': 1
}


def block_img(im, removed_datas):
    resized_ratio = 4.0
    im_height, im_width, dim = im.shape
    removed_mask = np.zeros((int(im_height/resized_ratio),
        int(im_width/resized_ratio)), np.uint8)
    for i, box in enumerate(removed_datas):
        new_box = [(point[0]/resized_ratio, point[1]/resized_ratio) for point in box]
        poly = Polygon(new_box)
        if not poly.is_valid:
            return False
        minx = int(max(0, min([point[0] for point in new_box])))
        miny = int(max(0, min([point[1] for point in new_box])))
        maxx = int(min(im_width, max([point[0] for point in new_box])))
        maxy = int(min(im_height, max([point[1] for point in new_box])))
        for row in range(miny, maxy):
            for col in range(minx, maxx):
                p = SH_point((col, row))
                inter = p.intersection(poly)
                if inter.is_empty:
                    continue
                removed_mask[row,col] = 255
    removed_mask = cv2.resize(removed_mask, (im_width, im_height))
    im[removed_mask>125] = 128
    return True


def polygon_area(poly):
    """compute area of a polygon
    Args:
        poly [4, 2]
    Returns:
        float: area a the polygon
    """
    edge = [
        (poly[1][0] + poly[0][0]) * (poly[1][1] - poly[0][1]),
        (poly[2][0] + poly[1][0]) * (poly[2][1] - poly[1][1]),
        (poly[3][0] + poly[2][0]) * (poly[3][1] - poly[2][1]),
        (poly[0][0] + poly[3][0]) * (poly[0][1] - poly[3][1])
    ]
    return np.sum(edge)/2.


def check_polygons(img, word_polygons):
    validated_oriented_boxes = []
    for i in range(0, word_polygons.shape[0]):
        poly = word_polygons[i]
        cur_poly = geometry.Polygon(poly)
        #check whether the polygon is valid

        p_area = polygon_area(poly)
        if not cur_poly.is_valid:
            cur_poly = cur_poly.buffer(0)
            if cur_poly.area < 4:
                continue
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
            points = np.array(cur_poly.exterior.coords[:]).astype(np.int32)
            cv2.fillPoly(img, np.int32([points]), (128, 128, 128))
            continue
            #for j in range(0, 4):
            #    point0 = (points[j][0], points[j][1])
            #    point1 = (points[(j+1)%4][0], points[(j+1)%4][1])
            #    cv2.line(img, point0, point1, colors[j], 2)
            #outname = 'test.jpg'
            #cv2.imwrite(outname, img)
            #import pdb
            #pdb.set_trace()
        if abs(p_area) < 4:
            continue
        if p_area > 0:
            poly = poly[(0, 3, 2, 1), :]
        validated_oriented_boxes.append(np.expand_dims(poly, axis=0))
    if not validated_oriented_boxes:
        return None, None
    validated_oriented_boxes = np.concatenate(validated_oriented_boxes, axis=0)
    return img, validated_oriented_boxes

def create_synthtext_dataset(data_root, save_path_train, save_path_test,
        train_ratio=0.7, shuffle=True, n_max=None):
  """ Create tf records for the VGG SynthText dataset

  Args:
    data_root: the root folder for the datasets
    save_path_train: path to save the TF record
    save_path_test: path to save the TF record
    train_ratio: the ratio of samples to be used for training.
    list_name: list file name
    shuffle: bool, whether to shuffle examples
  """

  # load gt.mat
  print('Loading gt.mat ...')
  gt = sio.loadmat(os.path.join(data_root, 'gt.mat'))
  n_samples = gt['wordBB'].shape[1]
  print('Start writing to %s %s' % (save_path_train, save_path_test))
  writer_train = tf.python_io.TFRecordWriter(save_path_train)
  writer_test = tf.python_io.TFRecordWriter(save_path_test)

  if n_max is not None:
    n_samples = min(n_max, n_samples)

  if shuffle:
    indices = np.random.permutation(n_samples)
  else:
    indices = np.arange(n_samples)

  for i in tqdm(range(n_samples)):
    idx = indices[i]
    image_rel_path = str(gt['imnames'][0, idx][0])
    image_path = os.path.join(data_root, image_rel_path)
    # load image jpeg data
    with open(image_path, 'rb') as f:
      image_jpeg = f.read()
    nparr = np.fromstring(image_jpeg, np.uint8)
    img = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)

    annot_dict = {}
    annot_dict['folder'] = ''
    annot_dict['filename'] = image_rel_path
    annot_dict['image'] = img
    annot_dict['size'] = {
        'height': annot_dict['image'].shape[0],
        'width': annot_dict['image'].shape[1]}
    # word polygons
    word_polygons = gt['wordBB'][0, idx]
    if word_polygons.ndim == 2:
      word_polygons = np.expand_dims(word_polygons, axis=2)
    word_polygons = np.transpose(word_polygons, axes=[2,1,0])
    n_words = word_polygons.shape[0]

    img, word_polygons = check_polygons(img, word_polygons)
    if not isinstance(img, np.ndarray):
        continue
    word_polygons_flat = [float(o) for o in word_polygons.flatten()]

    objects = []
    for i, polygon in enumerate(word_polygons):
        minx = max(0, np.min(polygon[:, 0]))
        miny = max(0, np.min(polygon[:, 1]))
        maxx = min(img.shape[1], np.max(polygon[:, 0]))
        maxy = min(img.shape[0], np.max(polygon[:, 1]))
        cur_obj = {}
        cur_obj['difficult'] = 0
        cur_obj['oriented_box'] = polygon
        cur_obj['bndbox'] = {'xmin': minx,
                'ymin': miny, 'xmax': maxx, 'ymax': maxy}
        cur_obj['name'] = 'text'
        cur_obj['truncated'] = 0
        cur_obj['pose'] = ''
        objects.append(cur_obj)
    annot_dict['object'] = objects

    example = dict_to_tf_example(
      annot_dict, data_root , LABEL_MAP,
      image_subdirectory='')
    if random.random() < train_ratio:
        writer_train.write(example.SerializeToString())
    else:
        writer_test.write(example.SerializeToString())


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages'):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding text groundtruth fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  img_path = os.path.join(data['folder'], image_subdirectory, data['filename'])
  full_path = os.path.join(dataset_directory, img_path)
  #with tf.gfile.GFile(full_path, 'rb') as fid:
  #  encoded_jpg = fid.read()
  #encoded_jpg_io = io.BytesIO(encoded_jpg)
  #import pdb
  #pdb.set_trace()
  #image = PIL.Image.open(encoded_jpg_io)
  image = data['image']
  #to jpeg string
  encoded_jpg = cv2.imencode('.jpg', image)[1].tostring()
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  if False:
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
    img = np.array(image)
    for obj in data['object']:
      points = obj['oriented_box']
      for i, point in enumerate(points):
        cv2.circle(img, (int(point[0]), int(point[1])), 2, colors[i], -1)
    outname = 'test.jpg'
    cv2.imwrite(outname, img)
    import pdb
    pdb.set_trace()

  for l in data['object']:
      for i, p in enumerate(l['oriented_box']):
        l['oriented_box'][i] = (float(p[1])/height, float(p[0])/width)
  flattened_oriented_box = [p for l in data['object'] for i in l['oriented_box'] for p in i]

  for obj in data['object']:
    difficult = bool(int(obj['difficult']))
    if ignore_difficult_instances and difficult:
      continue

    difficult_obj.append(int(difficult))

    xmin.append(float(obj['bndbox']['xmin']) / width)
    ymin.append(float(obj['bndbox']['ymin']) / height)
    xmax.append(float(obj['bndbox']['xmax']) / width)
    ymax.append(float(obj['bndbox']['ymax']) / height)
    classes_text.append(obj['name'].encode('utf8'))
    classes.append(label_map_dict[obj['name']])
    truncated.append(int(obj['truncated']))
    poses.append(obj['pose'].encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/bbox/oriented_box': dataset_util.float_list_feature(flattened_oriented_box),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example




WORD_POLYGON_DIM = 8


def read_jpeg_check(image_path, forbid_grayscale=False):
  with open(image_path, 'rb') as f:
    image_jpeg = f.read()
  return image_jpeg
  # import imghdr
  # import numpy as np
  # # check path exists
  # if not os.path.exists(image_path):
  #   print('Image does not exist: {}'.format(image_path))
  #   return None
  # # check file not empty
  # with open(image_path, 'rb') as f:
  #   image_jpeg = f.read()
  # if image_jpeg is None:
  #   print('Image file is empty: {}'.format(image_path))
  #   return None
  # # check image type is jpeg
  # if imghdr.what(image_path) != 'jpeg':
  #   print('Image file is not jpeg: {}'.format(image_path))
  #   return None
  # # check image is decodable
  # image_buf = np.fromstring(image_jpeg, dtype=np.uint8)
  # image = cv2.imdecode(image_buf, cv2.IMREAD_UNCHANGED)
  # if image is None:
  #   print('Failed to decode image: {}'.format(image_path))
  # # check image is not zero-size
  # if image.shape[0] * image.shape[1] == 0:
  #   print('Image has zero size: {}'.format(image_path))
  #   return None
  # # check image is not grayscale
  # if forbid_grayscale:
  #   if image.ndim == 2 or image.shape[2] == 1:
  #     print('Image is gray-scale: {}'.format(image_path))
  #     return None
  # return image_jpeg


class DatasetCreator(object):
  def __init__(self, save_path):
    self.save_path = save_path
    self.example_indicies = None

  def _read_list(self):
    """
    Read image and groundtruth list.
    RETURN
      `image_paths`: list of image file paths
      `gt_paths`: list of groundtruth file paths
    """
    raise NotImplementedError

  def _read_image(self, image_path):
    return np.asarray(Image.open(image_path))

  def _read_image_binary(self, image_path):
    return read_jpeg_check(image_path, forbid_grayscale=True)

  def _parse_annotation(self, annot_file_path, image):
    """
    Parse groundtruth annotations.
    ARGS
      `annot_file_path`: annotation file path
    RETURN
      `annot_dict`: dictionary of groundtruth annotations
    """
    raise NotImplementedError

  def create_data(self, image_name, annot_dict, image):
      data = {}
      data['folder'] = ''
      data['filename'] = image_name
      data['size'] = {
              'height': image.shape[0],
              'width': image.shape[1]
        }
      data['object'] = []
      word_polygons = annot_dict['word_polygons']
      objects = []
      for i, polygon in enumerate(word_polygons):
          minx = max(0, min([point[0] for point in polygon]))
          miny = max(0, min([point[1] for point in polygon]))
          maxx = min(image.shape[1], max([point[0] for point in polygon]))
          maxy = min(image.shape[0], max([point[1] for point in polygon]))
          cur_obj = {}
          cur_obj['difficult'] = 0
          cur_obj['bndbox'] = {'xmin': minx,
                  'ymin': miny, 'xmax': maxx, 'ymax': maxy}
          cur_obj['name'] = 'text'
          cur_obj['truncated'] = 0
          cur_obj['pose'] = ''
          objects.append(cur_obj)
      data['object'] = objects
      return data


  def _make_sample(self, image_name, annot_dict):
    """
    Make a protobuf example.
    ARGS
      `image_binaries`: str, image jpeg binaries
      `annot_dict`: dict, annotations
    RETURN
      `example`: protobuf example
    """
    if annot_dict['image'] is None:
      example = None
    else:
      annot_dict['folder'] = ''
      annot_dict['filename'] = image_name
      annot_dict['size'] = {
          'height': annot_dict['image'].shape[0],
          'width': annot_dict['image'].shape[1]}
      example = dict_to_tf_example(
        annot_dict, self.data_root , LABEL_MAP,
        image_subdirectory=self.subdirectory)
    return example

  def _create_next_sample(self):
    # initialize index
    if not hasattr(self, 'indices'):
      if self.shuffle:
        self.indices = np.random.permutation(self.n_samples)
      else:
        self.indices = np.arange(self.n_samples)
      self.index = 0

    # create the next sample if it's valid
    example = None
    if self.index < self.n_samples:
      image_path = self.image_paths[self.index]
      gt_path = self.gt_paths[self.index] if self.gt_paths is not None else None
      image_name = image_path.split('/')[-1]
      image = self._read_image(image_path)
      image.setflags(write=1)
      annot_dict = self._parse_annotation(gt_path, image)
      example = self._make_sample(image_name, annot_dict)
      self.index += 1

    return example

  def create(self):
    self._read_list()
    print('Start creating dataset with {} examples. Output path: {}'.format(
          self.n_samples, self.save_path))
    writer = tf.python_io.TFRecordWriter(self.save_path)
    count = 0
    for i in range(self.n_samples):
      example = self._create_next_sample()
      if example is not None:
        writer.write(example.SerializeToString())
        count += 1
      if i > 0 and i % 100 == 0:
        print('Progress %d / %d' % (i, self.n_samples))
    print('Done creating %d samples' % count)


class DatasetCreator_Icdar2015Incidental(DatasetCreator):
  def __init__(self, save_path, data_root, training=True, shuffle=True):
    self.save_path = save_path
    self.data_root = data_root
    self.training = training
    if training:
        self.subdirectory = 'Challenge4_training_set'
    else:
        self.subdirectory = 'Challenge4_testing_set'
    self.shuffle = shuffle

  def _read_list(self):
    if self.training:
      image_dir = os.path.join(self.data_root, 'Challenge4_training_set')
      gt_dir = os.path.join(self.data_root, 'Challenge4_training_gt')
    else:
      image_dir = os.path.join(self.data_root, 'Challenge4_testing_set')
      gt_dir = None

    self.image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
    if self.shuffle:
      random.shuffle(self.image_paths)
    if gt_dir is not None:
      self.gt_paths = [os.path.join(gt_dir, 'gt_{}.txt'.format(
          os.path.basename(o)[:-4])) for o in self.image_paths]
    else:
      self.gt_paths = None

    self.n_samples = len(self.image_paths)

  def _parse_annotation(self, gt_path, image):
    if gt_path is None:
      empty_data = {}
      empty_data['folder'] = ''
      empty_data['object'] = []
      return empty_data

    with io.open(gt_path, 'r', encoding='utf-8-sig') as f:
      lines = [o.strip() for o in f.readlines()]
    word_polygons = []
    remove_polygons = [] #we remove don't care text from the image.
    for line in lines:
      splits = line.split(',')
      polygon = [float(int(o)) for o in splits[:8]]
      points = [(polygon[2*i], polygon[2*i+1]) for i in range(0,4)]
      if splits[-1][0] == '#':
        remove_polygons.append(points)
      else:
        word_polygons.append(points)
    block_img(image, remove_polygons)

    data = {}
    data['folder'] = ''
    data['image'] = image
    objects = []
    for i, polygon in enumerate(word_polygons):
        minx = max(0, min([point[0] for point in polygon]))
        miny = max(0, min([point[1] for point in polygon]))
        maxx = min(image.shape[1], max([point[0] for point in polygon]))
        maxy = min(image.shape[0], max([point[1] for point in polygon]))
        cur_obj = {}
        cur_obj['difficult'] = 0
        cur_obj['bndbox'] = {'xmin': minx,
                'ymin': miny, 'xmax': maxx, 'ymax': maxy}
        cur_obj['oriented_box'] = polygon
        cur_obj['name'] = 'text'
        cur_obj['truncated'] = 0
        cur_obj['pose'] = ''
        objects.append(cur_obj)
    data['object'] = objects
    return data


class DatasetCreator_Icdar2013(DatasetCreator):
  def __init__(self, save_path, data_root, training, shuffle=False):
    self.save_path = save_path
    self.data_root = data_root
    self.training = training
    if training:
        self.subdirectory = 'Challenge2_Training_Task12_Images'
    else:
        self.subdirectory = 'Challenge2_Training_Task12_Images'
    self.shuffle = shuffle

  def _read_list(self):
    if self.training:
      image_dir = os.path.join(self.data_root, 'Challenge2_Training_Task12_Images')
      gt_dir = os.path.join(self.data_root, 'Task2.1_Training_GroundTrue')
    else:
      image_dir = os.path.join(self.data_root, 'Challenge2_Test_Task12_Images')
      gt_dir = os.path.join(self.data_root, 'Challenge2_Test_Task1_GT')

    # load image and groundtruth file list
    image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
    if self.shuffle:
      random.shuffle(image_paths)
    gt_paths = []
    for image_path in image_paths:
      image_id, _ = os.path.splitext(os.path.basename(image_path))
      gt_path = os.path.join(gt_dir, 'gt_%s.txt' % image_id)
      gt_paths.append(gt_path)

    self.image_paths = image_paths
    self.gt_paths = gt_paths
    self.n_samples = len(image_paths)

  def _parse_annotation(self, annot_file_path, image):
    with open(annot_file_path, 'r') as f:
      lines = [o.strip() for o in f.readlines()]
    p = re.compile('(\d+)[,\s]*?(\d+)[,\s]*?(\d+)[,\s]*?(\d+)[,\s]*?"(.*?)"')
    word_polygons = []
    for line in lines:
      m = p.match(line)
      xmin = int(m.group(1))
      ymin = int(m.group(2))
      xmax = int(m.group(3))
      ymax = int(m.group(4))
      # convert bounding box to polygon
      points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
      word_polygons.append(points)

    data = {}
    data['folder'] = ''
    data['image'] = image
    objects = []
    for i, polygon in enumerate(word_polygons):
        minx = max(0, min([point[0] for point in polygon]))
        miny = max(0, min([point[1] for point in polygon]))
        maxx = min(image.shape[1], max([point[0] for point in polygon]))
        maxy = min(image.shape[0], max([point[1] for point in polygon]))
        cur_obj = {}
        cur_obj['difficult'] = 0
        cur_obj['bndbox'] = {'xmin': minx,
                'ymin': miny, 'xmax': maxx, 'ymax': maxy}
        cur_obj['oriented_box'] = polygon
        cur_obj['name'] = 'text'
        cur_obj['truncated'] = 0
        cur_obj['pose'] = ''
        objects.append(cur_obj)
    data['object'] = objects
    return data


class DatasetCreator_UberText(DatasetCreator):
  def __init__(self, save_path, data_root, training, shuffle=False):
    self.save_path = save_path
    self.data_root = data_root
    self.training = training
    if training:
        self.subdirectory = 'train/1Kx1K/'
    else:
        self.subdirectory = 'val/1Kx1K/'
    self.shuffle = shuffle

  def _read_list(self):
    if self.training:
      image_dir = os.path.join(self.data_root, 'train/1Kx1K/')
      gt_dir = os.path.join(self.data_root, 'train/1Kx1K/')
    else:
      image_dir = os.path.join(self.data_root, 'val/1Kx1K/')
      gt_dir = os.path.join(self.data_root, 'val/1Kx1K/')

    # load image and groundtruth file list
    image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
    # We current do not consider blurred image in training
    image_paths = [path for path in image_paths if 'blurred' not in path]
    if self.shuffle:
      random.shuffle(image_paths)
    if gt_dir is not None:
      gt_paths = []
      for image_path in image_paths:
        image_id, _ = os.path.splitext(os.path.basename(image_path))
        gt_path = os.path.join(gt_dir, 'truth_%s.txt' % image_id)
        gt_paths.append(gt_path)
    self.image_paths = image_paths
    self.gt_paths = gt_paths
    self.n_samples = len(image_paths)

  def _parse_annotation(self, annot_file_path, image):
    if annot_file_path is None:
      return None
    with open(annot_file_path, 'r') as f:
      lines = [o.strip() for o in f.readlines()]
    p = re.compile('(\d+)[,\s]*?(\d+)[,\s]*?(\d+)[,\s]*?(\d+)[,\s]*?"(.*?)"')
    word_polygons = []
    with open(annot_file_path, 'r') as f:
      lines = [o.decode('utf-8-sig').encode('utf-8').strip() for o in f.readlines()]
      for line in lines:
        splits = line.split('\t')
        polygon = [float(int(o)) for o in splits[0].split(' ')]
        points = np.array([(polygon[i*2], polygon[i*2+1]) for i in range(len(polygon)/2)]).astype(np.float32)
        rect = cv2.minAreaRect(points)
        box = list(cv2.cv.BoxPoints(rect))
        text_type = splits[-1]
        word_polygons.append(box)

    data = {}
    data['folder'] = ''
    data['image'] = image
    objects = []
    for i, polygon in enumerate(word_polygons):
        minx = max(0, min([point[0] for point in polygon]))
        miny = max(0, min([point[1] for point in polygon]))
        maxx = min(image.shape[1], max([point[0] for point in polygon]))
        maxy = min(image.shape[0], max([point[1] for point in polygon]))
        cur_obj = {}
        cur_obj['difficult'] = 0
        cur_obj['bndbox'] = {'xmin': minx,
                'ymin': miny, 'xmax': maxx, 'ymax': maxy}
        cur_obj['oriented_box'] = polygon
        cur_obj['name'] = 'text'
        cur_obj['truncated'] = 0
        cur_obj['pose'] = ''
        objects.append(cur_obj)
    data['object'] = objects
    return data


def create_merge_multiple(save_path, creators, shuffle=True):
  n_sample_total = 0
  creator_indices = []
  for i, creator in enumerate(creators):
    creator._read_list()
    n_sample_total += creator.n_samples
    creator_indices.append(np.full((creator.n_samples), i, dtype=np.int))
  creator_indices = np.concatenate(creator_indices)

  if shuffle:
    np.random.shuffle(creator_indices)

  print('Start creating dataset with {} examples. Output path: {}'.format(
        n_sample_total, save_path))
  writer = tf.python_io.TFRecordWriter(save_path)
  count = 0
  for i in range(n_sample_total):
    creator = creators[creator_indices[i]]
    example = creator._create_next_sample()
    if example is not None:
      writer.write(example.SerializeToString())
      count += 1
    if i > 0 and i % 100 == 0:
      print('Progress %d / %d' % (i, n_sample_total))
  print('Done creating %d samples' % count)


if __name__ == '__main__':
  # ICDAR 2015 incidental
  ic15_data_root = '/path/to/ICDAR_2015/data/'
  save_path = '/path/to/save/tfrecord/file/'
  creator_ic15_train = DatasetCreator_Icdar2015Incidental(
      save_path=save_path,
      data_root=ic15_data_root,
      training=True,
      shuffle=True)
  #creator_ic15_train.create()
  #creator_ic15_test = DatasetCreator_Icdar2015Incidental(
  #    '/data_giles/duh188/git_tensorflow/models/research/object_detection/data/sceneText/icdar_2015_incidental_test_fasterrcnn.tf',
  #    ic15_data_root,
  #    training=False,
  #    shuffle=False)
  #creator_ic15_test.create()

  # ICDAR 2013
  #ic13_root_dir = '/data_giles/data/ICDAR/'
  #creator_ic13_train = DatasetCreator_Icdar2013('/data_giles/duh188/git_tensorflow/SceneText/text_detection/data/training/icdar_2013_training.tf',
  #    ic13_root_dir,
  #    training=True, shuffle=True)
  #creator_ic13_train.create()
  #creator_ic13_test = DatasetCreator_Icdar2013('../data/icdar_2013_test.tf',
  #    os.path.join(ic13_root_dir, 'Ch2_Scenet_Text/Text Localization'),
  #    training=False, shuffle=False)
