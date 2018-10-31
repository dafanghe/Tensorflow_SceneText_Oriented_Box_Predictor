This is an oriented object detector based on [tensorflow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).
Most of the code is not changed except those related to the need of predicinting oriented bounding boxes rather than regular horizontal bounding boxes.
Many tasks need to predict an oriented bounding box, e.g: Scene Text Detection.

# The reason of adopting this framework:
### Highly modular designed code
It's easy to change the encoding scheme in the code. Simply changing the code in box_coders folder.
The encoding using [R2CNN] (https://arxiv.org/abs/1706.09579) will be released soon.

### Natural integration with slim nets
It's easy to change feature extraction CNN backbone by using slim nets.

### Easy and clear configuration setting with google protobuf
Changing the network configuration setting is easy. For example, to change the different aspect ratios of the anchors used, simply changing the grid_anchor_generator in the configuration file.

### Many supporting codes have been provided.
It provides many supporting code such as exporting the trained model to a frozen graph that can be used in production(For example, in your c++ project).
Check out my another project [DeepSceneTextReader](https://github.com/dafanghe/DeepSceneTextReader) which used the frozen graph trained with this code.
