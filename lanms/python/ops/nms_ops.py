from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader


_nms_so_path = resource_loader.get_path_to_datafile("_nms_ops.so")
_locality_aware_nms_ops = load_library.load_op_library(_nms_so_path)
locality_aware_nms = _locality_aware_nms_ops.locality_aware_nms
