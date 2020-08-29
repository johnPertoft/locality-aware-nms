# Locality Aware NMS
- An implementation of Locality Aware NMS as described 
in [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155) 
as a custom Tensorflow OP.
- A standard NMS implementation as a custom Tensorflow OP.
- Support for rotated bounding boxes.
- Usable with Tensorflow 1.X and 2.
- Usable with Tensorflow Serving for production purposes.

## Usage
```python
from lanms import locality_aware_nms

# vertices: Tensor of shape (?, 4, 2).
# probs: Tensor of shape (?,).

vertices, scores = locality_aware_nms(vertices, probs, iou_threshold=0.3)

# vertices: Tensor of shape (?, 4, 2).
# scores: Tensor of shape (?,).
```

## Installation
With the current setup, the installed python package only works if it was built with the same Tensorflow
version as it's being used with. I didn't look further into this problem so I'm not sure what causes it
or how it can be fixed. Thus, in order to install this, it must first be built with the same python and 
tensorflow version as you will use it with. These versions can be passed to the build script as follows.

```
./build.sh 3.6 2.3.0
pip install artifacts/tf_locality_aware_nms-0.0.1-cp36-cp36m-linux_x86_64.whl
```

## Serving
As far as I know the built shared library file can not be dynamically loaded by Tensorflow Serving.
Instead Tensorflow Serving has to be built with these ops included in order to load a graph containing them. This can be achieved by the following steps before building Tensorflow Serving.

**Include in Tensorflow Serving repository**
```
cp -r lanms $TFSERVING/tensorflow_serving/custom_ops/
```

**Update the BUILD file to include these ops (as of version 2.1)**
```
sed -i '/SUPPORTED_TENSORFLOW_OPS =/a \ "//tensorflow_serving/custom_ops/lanms:nms_ops",' $TFSERVING/tensorflow_serving/model_servers/BUILD
```


## Also see
- [Official(?) C++ Locality-Aware NMS implementation](https://github.com/argman/EAST/blob/master/lanms/lanms.h)
- [(Slower) Pure Tensorflow 1.X implementation of Locality-Aware NMS](https://gist.github.com/johnPertoft/4b909fd099b60df01a041cd98f17a1dc)

## TODO
- I'm not convinced that the way the weighted merge is implemented is correct / the best way to do it.
  Given some order of bounding boxes (arbitrarily determined by row of top most coordinate) and assuming 
  similarly scored bounding boxes the current implementation will produce a final bounding box closer
  to the earlier ones rather than a bounding box that appears to be in the middle of a group of boxes.
