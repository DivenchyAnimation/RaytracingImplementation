#include "BoundingBox.h"
#include "pch.h"

struct Vertex {
  float x, y, z;
  float nx, ny, nz;
  double r, g, b;
};

struct Triangle {
  Vertex v1, v2, v3;
  double r, g, b;
  BoundingBox bbox;
};

void makeTriangles(std::vector<Triangle> &triangles,
                   const std::vector<float> &posBuf,
                   const std::vector<float> &norBuf, int task);
void meshTo2D(std::vector<float> &posBuf, BoundingBox &meshBBox, int width,
              int height);
void computeBarycentricCoords(const Vertex &v1, const Vertex &v2,
                              const Vertex &v3, int px, int py, float &alpha,
                              float &beta, float &gamma);
