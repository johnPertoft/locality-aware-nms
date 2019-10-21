#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

#include "nms.h"

using namespace tensorflow;

static inline void
_check_input_iou_threshold(OpKernelContext* context, const float iou_threshold) {
  OP_REQUIRES(context, iou_threshold >= 0 && iou_threshold <= 1,
      errors::InvalidArgument("iou_threshold must be in [0, 1]"));
}

static inline void
_check_input_bounding_boxes(OpKernelContext* context, const Tensor &vertices, const Tensor &probs) {
  OP_REQUIRES(context, vertices.dims() == 3,
        errors::InvalidArgument("vertices must be 3-D", vertices.shape().DebugString()));
  OP_REQUIRES(context, vertices.dim_size(1) == 4 && vertices.dim_size(2) == 2,
      errors::InvalidArgument("vertices must be shape (?, 4, 2)"));

  OP_REQUIRES(context, probs.dims() == 2,
      errors::InvalidArgument("probs must be 2-D", probs.shape().DebugString()));
  OP_REQUIRES(context, probs.dim_size(1) == 1,
      errors::InvalidArgument("probs must be shape (?, 1)"));
}

float
_get_input_iou_threshold(OpKernelContext* context) {
  const float iou_threshold = context->input(2).scalar<float>()();
  _check_input_iou_threshold(context, iou_threshold);
  return iou_threshold;
}

std::vector<nms::BoundingBox>
_get_input_bounding_boxes(OpKernelContext* context) {
  const Tensor& vertices = context->input(0);
  const Tensor& probs = context->input(1);
  _check_input_bounding_boxes(context, vertices, probs);

  std::vector<nms::BoundingBox> bounding_boxes;
  auto n = vertices.shape().dim_size(0);
  auto vertices_data = vertices.tensor<float, 3>();
  auto probs_data = probs.tensor<float, 2>();
  for (std::size_t i = 0; i < n; i++) {
    bounding_boxes.push_back(nms::BoundingBox{
      {{vertices_data(i, 0, 0), vertices_data(i, 0, 1)},
       {vertices_data(i, 1, 0), vertices_data(i, 1, 1)},
       {vertices_data(i, 2, 0), vertices_data(i, 2, 1)},
       {vertices_data(i, 3, 0), vertices_data(i, 3, 1)}},
      probs_data(i, 0)
    });
  }

  return bounding_boxes;
}

void
_populate_output_tensors(OpKernelContext* context, const std::vector<nms::BoundingBox> &bounding_boxes) {
  // Allocate output tensor for vertices.
  Tensor* vertices_output = NULL;
  TensorShape vertices_output_shape({int(bounding_boxes.size()), 4, 2});
  OP_REQUIRES_OK(context, context->allocate_output(0, vertices_output_shape, &vertices_output));

  // Allocate output tensor for scores.
  Tensor* scores_output = NULL;
  TensorShape scores_output_shape({int(bounding_boxes.size())});
  OP_REQUIRES_OK(context, context->allocate_output(1, scores_output_shape, &scores_output));

  // Copy data into output tensors.
  auto vertices_output_data = vertices_output->tensor<float, 3>();
  auto scores_output_data = scores_output->tensor<float, 1>();
  for (std::size_t i = 0; i < bounding_boxes.size(); i++) {
    for (std::size_t j = 0; j < 4; j++) {
      vertices_output_data(i, j, 0) = bounding_boxes[i].poly[j].x;
      vertices_output_data(i, j, 1) = bounding_boxes[i].poly[j].y;
    }

    scores_output_data(i) = bounding_boxes[i].score;
  }
}

class LocalityAwareNMSOp : public OpKernel {
 public:
  explicit LocalityAwareNMSOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const float iou_threshold = _get_input_iou_threshold(context);
    std::vector<nms::BoundingBox> bounding_boxes = _get_input_bounding_boxes(context);
    std::vector<nms::BoundingBox> merged_bounding_boxes = nms::locality_aware_nms(bounding_boxes, iou_threshold);
    _populate_output_tensors(context, merged_bounding_boxes);
  }
};

REGISTER_KERNEL_BUILDER(Name("LocalityAwareNMS").Device(DEVICE_CPU), LocalityAwareNMSOp);

class StandardNMSOp : public OpKernel {
 public:
  explicit StandardNMSOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const float iou_threshold = _get_input_iou_threshold(context);
    std::vector<nms::BoundingBox> bounding_boxes = _get_input_bounding_boxes(context);
    std::vector<nms::BoundingBox> merged_bounding_boxes = nms::standard_nms(bounding_boxes, iou_threshold);
    _populate_output_tensors(context, merged_bounding_boxes);
  }
};

REGISTER_KERNEL_BUILDER(Name("StandardNMS").Device(DEVICE_CPU), StandardNMSOp);
