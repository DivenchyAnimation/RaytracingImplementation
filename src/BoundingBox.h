#include "pch.h"

struct BoundingBox {
  float minx, miny, minz, maxx, maxy, maxz;
};

BoundingBox computeTriBBox(const float x1, const float x2, const float x3,
                           const float y1, const float y2, const float y3);
BoundingBox meshBoundBox(const std::vector<float> &posBuf);
