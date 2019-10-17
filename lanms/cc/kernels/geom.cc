#include <cmath>
#include <cstddef>
#include <vector>

#include "geom.h"


namespace geom {

/*
Note:
In these functions we assume that
- polygons are ordered in clockwise order.
- polygons don't have more than two collinear vertices.
- polygons have at least three vertices.
- polygons are convex.
*/

float
polygon_area(const Polygon &polygon) {
  // Return the area of the polygon.
  float area = 0.0;
  for (std::size_t i = 0; i < polygon.size(); i++) {
    auto j = (i + 1) % polygon.size();
    area += polygon[i].x * polygon[j].y - polygon[j].x * polygon[i].y;
  }
  area = area / 2.0;
  area = std::fabs(area);
  return area;
}

Point
compute_intersection(const Point &p1, const Point &p2, const Point &v1, const Point &v2) {
  // Computes the intersection point of the line segment p1 -> p2 and the infinite edge v1 -> v2.
  using Vec2 = Point;
  auto dc = Vec2({v1.x - v2.x, v1.y - v2.y});
  auto dp = Vec2({p2.x - p1.x, p2.y - p1.y});
  float n1 = v1.x * v2.y - v1.y * v2.x;
  float n2 = p2.x * p1.y - p2.y * p1.x;
  float n3 = 1.0 / (dc.x * dp.y - dc.y * dp.x);
  return Point({
    (n1 * dp.x - n2 * dc.x) * n3,
    (n1 * dp.y - n2 * dc.y) * n3
  });
}

bool
inside_edge(const Point &p, const Point &v1, const Point &v2) {
  // Return whether the point p is inside of (right of) the edge v1 -> v2.
  return (v2.x - v1.x) * (p.y - v1.y) > (v2.y - v1.y) * (p.x - v1.x);
}

Polygon
polygon_intersection(const Polygon &subject_polygon, const Polygon &clip_polygon) {
  // Implements the Sutherland-Hodgman algorithm for polygon clipping.
  // See https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm

  // Initial polygon.
  Polygon intersection_polygon = subject_polygon;

  // Iterate over clip edges.
  for (std::size_t i = 0; i < clip_polygon.size(); i++) {
    Polygon current_polygon = intersection_polygon;
    intersection_polygon.clear();

    auto j = (i + 1) % clip_polygon.size();
    Point v1 = clip_polygon[i];
    Point v2 = clip_polygon[j];

    // Iterate over the points in the current polygon.
    for (std::size_t k = 0; k < current_polygon.size(); k++) {
      Point current_point = current_polygon[k];
      Point prev_point = current_polygon[(k + current_polygon.size() - 1) % current_polygon.size()];

      Point intersecting_point = compute_intersection(prev_point, current_point, v1, v2);

      if (inside_edge(current_point, v1, v2)) {
        if (!inside_edge(prev_point, v1, v2)) {
          intersection_polygon.push_back(intersecting_point);
        }
        intersection_polygon.push_back(current_point);
      } else if (inside_edge(prev_point, v1, v2)) {
        intersection_polygon.push_back(intersecting_point);
      }
    }
  }

  return intersection_polygon;
}

float
intersection_over_union(const Polygon &a, const Polygon &b) {
  // Return the ratio of the areas of the intersection and union of polygons a and b.
  auto intersection_area = polygon_area(polygon_intersection(a, b));
  auto union_area = polygon_area(a) + polygon_area(b) - intersection_area;
  auto iou = intersection_area / union_area;
  return iou;
}

}
