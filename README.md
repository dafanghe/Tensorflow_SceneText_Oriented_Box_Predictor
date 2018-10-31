This is an oriented object detector based on [tensorflow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).
Most of the code is not changed except those related to the need of predicinting oriented bounding boxes rather than regular horizontal bounding boxes.
Many tasks need to predict an oriented bounding box, e.g: Scene Text Detection.

# Goals
For each predicted bounding boxes, in addition to the regular horizontal bounding box, we need to predict one oriented bounding box.
Basically it means that we need to regress to an oriented bounding box.
In this project, we simply regress to the encoded 4 corners of the oriented bounding boxes(8 values).
See below equation for the encoding function. j is the index for each corner. g represents ground truth oriented bounding boxes.
w_a and h_a is the anchor width and height, respectively.

<p align="center">
  <img src="pics/encoding.png" height=120>
</p>

# The reason of adopting this Faster RCNN/SSD framework:
There are many object detection framework to be used. We adopt this one as the basis for the following reasons:
### Highly modular designed code
It's easy to change the encoding scheme in the code. Simply changing the code in box_coders folder.
The encoding using [R2CNN] (https://arxiv.org/abs/1706.09579) will be released soon.
Training model with faster rcnn or ssd is easy to modify.

### Natural integration with slim nets
It's easy to change feature extraction CNN backbone by using slim nets.

### Easy and clear configuration setting with google protobuf
Changing the network configuration setting is easy. For example, to change the different aspect ratios of the anchors used, simply changing the grid_anchor_generator in the configuration file.

### Many supporting codes have been provided.
It provides many supporting code such as exporting the trained model to a frozen graph that can be used in production(For example, in your c++ project).
Check out my another project [DeepSceneTextReader](https://github.com/dafanghe/DeepSceneTextReader) which used the frozen graph trained with this code.

# Code Changed

### proto files
added oriented related filed to the proto files. Please build them with

```
protoc protos/*.proto --python_out=.
```

### Box encoding scheme
added code for encode oriented bounding boxes
