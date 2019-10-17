#ifndef GEOM_H_
#define GEOM_H_

#include <vector>


namespace geom {

struct Point {
  float x;
  float y;
};

typedef std::vector<Point> Polygon;

float
polygon_area(const Polygon &polygon);

Point
compute_intersection(const Point &p1, const Point &p2, const Point &v1, const Point &v2);

bool
inside_edge(const Point &p, const Point &v1, const Point &v2);

Polygon
polygon_intersection(const Polygon &subject_polygon, const Polygon &clip_polygon);

float
intersection_over_union(const Polygon &a, const Polygon &b);

}

#endif