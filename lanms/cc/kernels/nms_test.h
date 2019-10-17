#include <cstddef>
#include <vector>

#include "gtest/gtest.h"

#include "nms.h"


TEST(min_y, simple_square) {
  nms::BoundingBox b{
    {{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}},
    1.0
  };
  EXPECT_FLOAT_EQ(0.0, nms::min_y(b));
}

TEST(min_y, rotated_square) {
  nms::BoundingBox b{
    {{150.0, 79.0}, {221.0, 150.0}, {150.0, 221.0}, {79.0, 150.0}},
    1.0
  };
  EXPECT_FLOAT_EQ(79.0, nms::min_y(b));
}

TEST(should_merge, square_with_itself) {
  nms::BoundingBox b{
    {{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}},
    1.0
  };
  EXPECT_TRUE(nms::should_merge(b, b, 0.1));
  EXPECT_TRUE(nms::should_merge(b, b, 0.3));
  EXPECT_TRUE(nms::should_merge(b, b, 0.5));
  EXPECT_TRUE(nms::should_merge(b, b, 1.0));
}

TEST(should_merge, two_squares) {
  nms::BoundingBox b1{
    {{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}},
    1.0
  };

  nms::BoundingBox b2{
    {{2.0, 0.0}, {12.0, 0.0}, {12.0, 10.0}, {2.0, 10.0}},
    1.0
  };

  float iou = 8.0 / 12.0;
  EXPECT_TRUE(nms::should_merge(b1, b2, iou - 0.1));
  EXPECT_FALSE(nms::should_merge(b1, b2, iou + 0.1));
}

TEST(weighted_merge, square_with_itself) {
  nms::BoundingBox b{
    {{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}},
    1.0
  };
  auto mb = nms::weighted_merge(b, b);
  ASSERT_EQ(4, mb.poly.size());
  for (std::size_t i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(b.poly[i].x, mb.poly[i].x);
    EXPECT_FLOAT_EQ(b.poly[i].y, mb.poly[i].y);
  }
  EXPECT_FLOAT_EQ(2.0, mb.score);
}

TEST(weighted_merge, two_squares_equal_scores) {
  nms::BoundingBox b1{
    {{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}},
    1.0
  };

  nms::BoundingBox b2{
    {{5.0, 0.0}, {15.0, 0.0}, {15.0, 10.0}, {5.0, 10.0}},
    1.0
  };

  auto mb = nms::weighted_merge(b1, b2);
  ASSERT_EQ(4, mb.poly.size());
  for (std::size_t i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ((b1.poly[i].x + b2.poly[i].x) / 2.0, mb.poly[i].x);
    EXPECT_FLOAT_EQ((b1.poly[i].y + b2.poly[i].y) / 2.0, mb.poly[i].y);
  }
  EXPECT_FLOAT_EQ(2.0, mb.score);
}

TEST(weighted_merge, two_square_non_equal_scores) {
  nms::BoundingBox b1{
    {{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}},
    1.0
  };

  nms::BoundingBox b2{
    {{5.0, 0.0}, {15.0, 0.0}, {15.0, 10.0}, {5.0, 10.0}},
    0.5
  };

  auto mb = nms::weighted_merge(b1, b2);
  ASSERT_EQ(4, mb.poly.size());
  for (std::size_t i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ((1.0 * b1.poly[i].x + 0.5 * b2.poly[i].x) / 1.5, mb.poly[i].x);
    EXPECT_FLOAT_EQ((1.0 * b1.poly[i].y + 0.5 * b2.poly[i].y) / 1.5, mb.poly[i].y);
  }
  EXPECT_FLOAT_EQ(1.5, mb.score);
}

TEST(standard_nms, single_square) {
  nms::BoundingBox b{
    {{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}},
    1.0
  };
  std::vector<nms::BoundingBox> bounding_boxes{b};
  auto res = nms::standard_nms(bounding_boxes, 0.5);
  ASSERT_EQ(1, res.size());
  for (std::size_t i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(b.poly[i].x, res[0].poly[i].x);
    EXPECT_FLOAT_EQ(b.poly[i].y, res[0].poly[i].y);
  }
  EXPECT_FLOAT_EQ(1.0, res[0].score);
}

TEST(standard_nms, two_squares_with_overlap) {
  nms::BoundingBox b1{
    {{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}},
    1.0
  };

  nms::BoundingBox b2{
    {{2.0, 0.0}, {12.0, 0.0}, {12.0, 10.0}, {2.0, 10.0}},
    0.9
  };

  std::vector<nms::BoundingBox> bounding_boxes{b1, b2};
  auto res = nms::standard_nms(bounding_boxes, 0.5);
  ASSERT_EQ(1, res.size());
  for (std::size_t i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(b1.poly[i].x, res[0].poly[i].x);
    EXPECT_FLOAT_EQ(b1.poly[i].y, res[0].poly[i].y);
  }
  EXPECT_FLOAT_EQ(1.0, res[0].score);
}

TEST(standard_nms, two_squares_without_overlap) {
  nms::BoundingBox b1{
    {{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}},
    1.0
  };

  nms::BoundingBox b2{
    {{0.0, 20.0}, {10.0, 20.0}, {10.0, 30.0}, {0.0, 30.0}},
    0.9
  };

  std::vector<nms::BoundingBox> bounding_boxes{b1, b2};
  auto res = nms::standard_nms(bounding_boxes, 0.5);
  ASSERT_EQ(2, res.size());

  for (std::size_t i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(b1.poly[i].x, res[0].poly[i].x);
    EXPECT_FLOAT_EQ(b1.poly[i].y, res[0].poly[i].y);
  }
  EXPECT_FLOAT_EQ(1.0, res[0].score);

  for (std::size_t i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(b2.poly[i].x, res[1].poly[i].x);
    EXPECT_FLOAT_EQ(b2.poly[i].y, res[1].poly[i].y);
  }
  EXPECT_FLOAT_EQ(0.9, res[1].score);
}

// TODO: Should we add basically the same tests for lanms that we already have on python side?
//  or just make a comment about it.
