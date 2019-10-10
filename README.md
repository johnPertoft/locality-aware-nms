# Locality Aware NMS
This repository contains an implementation of Locality Aware NMS as described in [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155) as a custom Tensorflow OP.

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

