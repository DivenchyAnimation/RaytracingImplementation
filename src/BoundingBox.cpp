#include "BoundingBox.h"

using std::vector, std::min, std::max, std::numeric_limits;

BoundingBox meshBoundBox(const std::vector<float> &posBuf) {
  BoundingBox bbox;
  bbox.minx = bbox.miny = bbox.minz = numeric_limits<float>::max();
  bbox.maxx = bbox.maxy = bbox.maxz = numeric_limits<float>::min();

  for (size_t i = 0; i < posBuf.size(); i += 3) {

    if (posBuf[i] < bbox.minx) {
      bbox.minx = posBuf[i];
    }
    if (posBuf[i + 1] < bbox.miny) {
      bbox.miny = posBuf[i + 1];
    }
    if (posBuf[i + 2] < bbox.minz) {
      bbox.minz = posBuf[i + 2];
    }
    if (posBuf[i] > bbox.maxx) {
      bbox.maxx = posBuf[i];
    }
    if (posBuf[i + 1] > bbox.maxy) {
      bbox.maxy = posBuf[i + 1];
    }
    if (posBuf[i + 2] > bbox.maxz) {
      bbox.maxz = posBuf[i + 2];
    }
  }

  return bbox;
}

BoundingBox computeTriBBox(const float x1, const float x2, const float x3,
                           const float y1, const float y2, const float y3) {
  BoundingBox bbox;
  bbox.minx = min({x1, x2, x3});
  bbox.miny = min({y1, y2, y3});
  bbox.maxx = max({x1, x2, x3});
  bbox.maxy = max({y1, y2, y3});
  return bbox;
}
