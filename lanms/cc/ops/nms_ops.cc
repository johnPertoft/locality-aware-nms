#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("LocalityAwareNMS")
    .Input("vertices: float32")
    .Input("probs: float32")
    .Input("iou_threshold: float32")
    .Output("vertices_output: float32")
    .Output("scores_output: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim(), 4, 2}));
      c->set_output(1, c->MakeShape({c->UnknownDim()}));
      return Status::OK();
    });