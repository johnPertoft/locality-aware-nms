#ifndef NMS_H_
#define NMS_H_

#include <vector>

#include "geom.h"


namespace nms {

struct BoundingBox {
  geom::Polygon poly;
  float score;
};

float
min_y(const BoundingBox &b);

bool
should_merge(const BoundingBox &a, const BoundingBox &b, float iou_threshold);

BoundingBox
weighted_merge(const BoundingBox &a, const BoundingBox &b);

std::vector<BoundingBox>
standard_nms(const std::vector<BoundingBox> &bounding_boxes, float iou_threshold);

std::vector<BoundingBox>
locality_aware_nms(std::vector<BoundingBox> &bounding_boxes, float iou_threshold);

}

#endif
