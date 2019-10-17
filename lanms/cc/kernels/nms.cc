#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

#include "geom.h"
#include "nms.h"


namespace nms {

float
min_y(const BoundingBox &b) {
  auto y_min = b.poly[0].y;
  for (std::size_t i = 1; i < 4; i++) {
    if (b.poly[i].y < y_min) {
      y_min = b.poly[i].y;
    }
  }
  return y_min;
}

bool
should_merge(const BoundingBox &a, const BoundingBox &b, float iou_threshold) {
  return geom::intersection_over_union(a.poly, b.poly) >= iou_threshold;
}

BoundingBox
weighted_merge(const BoundingBox &a, const BoundingBox &b) {
  // Weighted merge as described in EAST paper.
  auto new_score = a.score + b.score;
  return BoundingBox{
    {{(a.score * a.poly[0].x + b.score * b.poly[0].x) / new_score, (a.score * a.poly[0].y + b.score * b.poly[0].y) / new_score},
     {(a.score * a.poly[1].x + b.score * b.poly[1].x) / new_score, (a.score * a.poly[1].y + b.score * b.poly[1].y) / new_score},
     {(a.score * a.poly[2].x + b.score * b.poly[2].x) / new_score, (a.score * a.poly[2].y + b.score * b.poly[2].y) / new_score},
     {(a.score * a.poly[3].x + b.score * b.poly[3].x) / new_score, (a.score * a.poly[3].y + b.score * b.poly[3].y) / new_score}},
    new_score
  };
}

std::vector<BoundingBox>
standard_nms(const std::vector<BoundingBox> &bounding_boxes, float iou_threshold) {
  // Create a sorted (by descending scores) list of candidate indices.
  std::vector<std::size_t> candidate_indices(bounding_boxes.size());
  std::iota(candidate_indices.begin(), candidate_indices.end(), 0);
  std::sort(candidate_indices.begin(), candidate_indices.end(), [&](std::size_t i, std::size_t j) {
    return bounding_boxes[i].score > bounding_boxes[j].score;
  });

  std::vector<std::size_t> keep_indices;

  while (candidate_indices.size()) {
    std::size_t p = 0;
    auto current_index = candidate_indices[0];
    keep_indices.push_back(current_index);

    // Only keep indices of bounding boxes that are not too close to the current bounding box.
    for (std::size_t i = 1; i < candidate_indices.size(); i++) {
      if (!should_merge(bounding_boxes[current_index], bounding_boxes[candidate_indices[i]], iou_threshold)) {
        candidate_indices[p++] = candidate_indices[i];
      }
    }
    candidate_indices.resize(p);
  }

  std::vector<BoundingBox> bounding_boxes_to_keep;

  for (auto &&i : keep_indices) {
    bounding_boxes_to_keep.push_back(bounding_boxes[i]);
  }

  return bounding_boxes_to_keep;
}

std::vector<BoundingBox>
locality_aware_nms(std::vector<BoundingBox> &bounding_boxes, float iou_threshold) {
  // Implements the Locality-Aware NMS algorithm as described in EAST (https://arxiv.org/abs/1704.03155)

  // Sort bounding boxes row wise by sorting by their top most y coordinate.
  std::sort(bounding_boxes.begin(), bounding_boxes.end(), [](const BoundingBox &a, const BoundingBox &b) -> bool {
    return min_y(a) < min_y(b);
  });

  std::vector<BoundingBox> merged_bounding_boxes;
  BoundingBox current = bounding_boxes[0];

  for (std::size_t i = 1; i < bounding_boxes.size(); i++) {
    if (should_merge(current, bounding_boxes[i], iou_threshold)) {
      current = weighted_merge(current, bounding_boxes[i]);
    } else {
      merged_bounding_boxes.push_back(current);
      current = bounding_boxes[i];
    }
  }

  merged_bounding_boxes.push_back(current);
  merged_bounding_boxes = standard_nms(merged_bounding_boxes, iou_threshold);

  return merged_bounding_boxes;
}

}