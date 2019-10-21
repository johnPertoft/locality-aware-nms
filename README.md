# Locality Aware NMS
This repository contains 
- An implementation of Locality Aware NMS as described 
in [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155) 
as a custom Tensorflow OP.
- A standard NMS implementation as a custom Tensorflow OP.
- Both implementation support rotated bounding boxes.

## Usage
```python
from lanms import locality_aware_nms

# vertices: Tensor of shape (?, 4, 2).
# probs: Tensor of shape (?,).

vertices, scores = locality_aware_nms(vertices, probs, FLAGS.nms_iou_threshold)

# vertices: Tensor of shape (?, 4, 2).
# scores: Tensor of shape (?,).
```

## Installation
TODO

## Serving
As far as I know the built shared library file can not be dynamically loaded by Tensorflow Serving.
Instead Tensorflow Serving has to be built with these ops included in order to load a graph containing them.
This can be achieved by the following steps.

**Include source files in Tensorflow Serving repository**
```
cp -r lanms $TFSERVING/tensorflow_serving/custom_ops/
```

**Update the BUILD file to include these ops**
```
sed -i '/SUPPORTED_TENSORFLOW_OPS =/a \ "//tensorflow_serving/custom_ops/lanms:nms_ops",' $TFSERVING/tensorflow_serving/model_servers/BUILD
```

## Development
TODO

## TODO