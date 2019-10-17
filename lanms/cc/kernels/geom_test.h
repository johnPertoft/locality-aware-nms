#include <cmath>
#include <cstddef>
#include <vector>

#include "gtest/gtest.h"

#include "geom.h"


TEST(polygon_area_test, simple_triangle) {
  geom::Polygon p{{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}};
  EXPECT_FLOAT_EQ(50.0, geom::polygon_area(p));
}

TEST(polygon_area_test, simple_square) {
  geom::Polygon p{{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}};
  EXPECT_FLOAT_EQ(100.0, geom::polygon_area(p));
}

TEST(polygon_area_test, simple_square_negative_coordinates) {
  geom::Polygon p{{-10.0, -10.0}, {10.0, -10.0}, {10.0, 10.0}, {-10, 10.0}};
  EXPECT_FLOAT_EQ(400.0, geom::polygon_area(p));
}

TEST(polygon_area_test, octagon) {
  geom::Polygon p{
      {50.0, 0.0}, {150.0, 0.0},
      {200.0, 25.0}, {200.0, 75.0},
      {150.0, 100.0}, {50.0, 100.0},
      {0.0, 75.0}, {0.0, 25.0}};

  // The octagon is contained in 200 x 100 rectangle.
  float rectangle_area = 200.0 * 100.0;
  float triangle_cutoff_area = 50.0 * 25.0 / 2.0;
  float expected_area = rectangle_area - 4 * triangle_cutoff_area;

  EXPECT_FLOAT_EQ(expected_area, geom::polygon_area(p));
}

TEST(compute_intersection, horizontal_edge_vertical_line) {
  geom::Point p1{0.0, 0.0};
  geom::Point p2{10.0, 0.0};
  geom::Point v1{20.0, 10.0};
  geom::Point v2{20.0, 20.0};

  auto i = geom::compute_intersection(p1, p2, v1, v2);

  EXPECT_FLOAT_EQ(20.0, i.x);
  EXPECT_FLOAT_EQ(0.0, i.y);
}

TEST(compute_intersection, horizontal_edge_vertical_line_overlap) {
  geom::Point p1{0.0, 10.0};
  geom::Point p2{20.0, 10.0};
  geom::Point v1{10.0, 0.0};
  geom::Point v2{10.0, 20.0};

  auto i = geom::compute_intersection(p1, p2, v1, v2);

  EXPECT_FLOAT_EQ(10.0, i.x);
  EXPECT_FLOAT_EQ(10.0, i.y);
}

TEST(compute_intersection, parallel_edge_and_line) {
  geom::Point p1{0.0, 0.0};
  geom::Point p2{10.0, 0.0};
  geom::Point v1{0.0, 10.0};
  geom::Point v2{10.0, 10.0};

  auto i = geom::compute_intersection(p1, p2, v1, v2);

  EXPECT_TRUE(std::isnan(i.x) || std::isinf(i.x));
  EXPECT_TRUE(std::isnan(i.y) || std::isinf(i.y));
}

TEST(inside_edge, horizontal_edge) {
  geom::Point p1{5.0, 0.0};
  geom::Point p2{5.0, 15.0};
  geom::Point v1{0.0, 10.0};
  geom::Point v2{10.0, 10.0};

  EXPECT_FALSE(geom::inside_edge(p1, v1, v2));
  EXPECT_TRUE(geom::inside_edge(p2, v1, v2));

  // Edge in opposite direction.
  EXPECT_TRUE(geom::inside_edge(p1, v2, v1));
  EXPECT_FALSE(geom::inside_edge(p2, v2, v1));
}

TEST(inside_edge, vertical_edge) {
  geom::Point p1{5.0, 5.0};
  geom::Point p2{15.0, 5.0};
  geom::Point v1{10.0, 0.0};
  geom::Point v2{10.0, 10.0};

  EXPECT_TRUE(geom::inside_edge(p1, v1, v2));
  EXPECT_FALSE(geom::inside_edge(p2, v1, v2));

  // Edge in opposite direction.
  EXPECT_FALSE(geom::inside_edge(p1, v2, v1));
  EXPECT_TRUE(geom::inside_edge(p2, v2, v1));
}

TEST(inside_edge, diagonal_edge) {
  geom::Point p1{0.0, 0.0};
  geom::Point p2{10.0, 10.0};
  geom::Point v1{0.0, 10.0};
  geom::Point v2{10.0, 0.0};

  EXPECT_FALSE(geom::inside_edge(p1, v1, v2));
  EXPECT_TRUE(geom::inside_edge(p2, v1, v2));

  // Edge in opposite direction.
  EXPECT_TRUE(geom::inside_edge(p1, v2, v1));
  EXPECT_FALSE(geom::inside_edge(p2, v2, v1));
}

TEST(polygon_intersection, square_with_itself) {
  geom::Polygon p{{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}};

  auto p_intersection = geom::polygon_intersection(p, p);

  ASSERT_EQ(4, p_intersection.size());

  // Note: The order of points might be shifted but not permuted. We need to find the offset
  // from which to compare the points in the intersection polygon and the original polygon.
  auto first = p_intersection[0];
  int offset = -1;
  for (std::size_t i = 0; i < 4; i++) {
    // TODO: Should probably be float comparison.
    if (first.x == p[i].x && first.y == p[i].y) {
      offset = i;
    }
  }
  ASSERT_TRUE(offset >= 0);

  for (std::size_t i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(p_intersection[i].x, p[(i + offset) % 4].x);
    EXPECT_FLOAT_EQ(p_intersection[i].y, p[(i + offset) % 4].y);
  }
}

TEST(polygon_intersection, square_on_rotated_square) {
  geom::Polygon p1{{100.0, 100.0}, {200.0, 100.0}, {200.0, 200.0}, {100.0, 200.0}};
  geom::Polygon p2{{150.0, 79.0}, {221.0, 150.0}, {150.0, 221.0}, {79.0, 150.0}};
  EXPECT_EQ(8, geom::polygon_intersection(p1, p2).size());
}

TEST(intersection_over_union, square_with_itself) {
  geom::Polygon p{{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}};
  EXPECT_FLOAT_EQ(1.0, geom::intersection_over_union(p, p));
}

TEST(intersection_over_union, simple_squares) {
  geom::Polygon p1{{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}};
  geom::Polygon p2{{5.0, 0.0}, {15.0, 0.0}, {15.0, 10.0}, {5.0, 10.0}};
  EXPECT_FLOAT_EQ(0.5 / 1.5, geom::intersection_over_union(p1, p2));
}

TEST(intersection_over_union, squares_without_overlap) {
  geom::Polygon p1{{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}};
  geom::Polygon p2{{20.0, 0.0}, {20.0, 0.0}, {20.0, 10.0}, {20.0, 10.0}};
  EXPECT_FLOAT_EQ(0.0, geom::intersection_over_union(p1, p2));
}

TEST(intersection_over_union, square_inside_square) {
  geom::Polygon p1{{0.0, 0.0}, {100.0, 0.0}, {100.0, 100.0}, {0.0, 100.0}};
  geom::Polygon p2{{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}};
  EXPECT_FLOAT_EQ(100.0 / 10000.0, geom::intersection_over_union(p1, p2));
}
