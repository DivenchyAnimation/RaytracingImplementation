#include "Triangle.h"
#include "common.h"

using std::vector, std::min;

///// Helpers
void calcColors(int count, double &r, double &g, double &b) {
  r = 255 * RANDOM_COLORS[count % 7][0];
  g = 255 * RANDOM_COLORS[count % 7][1];
  b = 255 * RANDOM_COLORS[count % 7][2];
}

void generateTriangles(Vertex &v1, Vertex &v2, Vertex &v3,
                       const vector<float> &posBuf, const vector<float> &norBuf,
                       vector<Triangle> &triangles) {
  int count = 0;
  for (size_t i = 0; i < posBuf.size(); i += 9) {
    v1 = {posBuf[i], posBuf[i + 1], posBuf[i + 2],
          norBuf[i], norBuf[i + 1], norBuf[i + 2]};
    v2 = {posBuf[i + 3], posBuf[i + 4], posBuf[i + 5],
          norBuf[i + 3], norBuf[i + 4], norBuf[i + 5]};
    v3 = {posBuf[i + 6], posBuf[i + 7], posBuf[i + 8],
          norBuf[i + 6], norBuf[i + 7], norBuf[i + 8]};
    Triangle tri = {v1,
                    v2,
                    v3,
                    255 * RANDOM_COLORS[count % 7][0],
                    255 * RANDOM_COLORS[count % 7][1],
                    255 * RANDOM_COLORS[count % 7][2],
                    computeTriBBox(v1.x, v2.x, v3.x, v1.y, v2.y, v3.y)};
    triangles.push_back(tri);
    count++;
  }
}

void generateTrianglesVColor(Vertex &v1, Vertex &v2, Vertex &v3,
                             const vector<float> &posBuf,
                             const vector<float> &norBuf,
                             vector<Triangle> &triangles) {
  int count = 0;
  double r, g, b;
  for (size_t i = 0; i < posBuf.size(); i += 9) {
    calcColors(count, r, g, b);
    v1 = {posBuf[i],
          posBuf[i + 1],
          posBuf[i + 2],
          norBuf[i],
          norBuf[i + 1],
          norBuf[i + 2],
          r,
          g,
          b};
    count++;
    calcColors(count, r, g, b);
    v2 = {posBuf[i + 3],
          posBuf[i + 4],
          posBuf[i + 5],
          norBuf[i + 3],
          norBuf[i + 4],
          norBuf[i + 5],
          r,
          g,
          b};
    count++;
    calcColors(count, r, g, b);
    v3 = {posBuf[i + 6],
          posBuf[i + 7],
          posBuf[i + 8],
          norBuf[i + 6],
          norBuf[i + 7],
          norBuf[i + 8],
          r,
          g,
          b};
    count++;
    Triangle tri = {v1,
                    v2,
                    v3,
                    r,
                    g,
                    b,
                    computeTriBBox(v1.x, v2.x, v3.x, v1.y, v2.y, v3.y)};
    triangles.push_back(tri);
    count++;
  }
}

///// End of helpers

///// Method Definitions

// Determine Barycentric Coordinated
void computeBarycentricCoords(const Vertex &v1, const Vertex &v2,
                              const Vertex &v3, int px, int py, float &alpha,
                              float &beta, float &gamma) {
  float denominator =
      (v2.y - v3.y) * (v1.x - v3.x) + (v3.x - v2.x) * (v1.y - v3.y);
  alpha =
      ((v2.y - v3.y) * (px - v3.x) + (v3.x - v2.x) * (py - v3.y)) / denominator;
  beta =
      ((v3.y - v1.y) * (px - v3.x) + (v1.x - v3.x) * (py - v3.y)) / denominator;
  gamma = 1.0f - alpha - beta;
}

void makeTriangles(vector<Triangle> &triangles, const vector<float> &posBuf,
                   const vector<float> &norBuf, int task) {
  Vertex v1, v2, v3;
  switch (task) {
  case 3:
    generateTrianglesVColor(v1, v2, v3, posBuf, norBuf, triangles);
    break;
  default:
    generateTriangles(v1, v2, v3, posBuf, norBuf, triangles);
  }
}

// I believe I used ChatGPT to correct some error that was occuring in this
// function
void meshTo2D(vector<float> &posBuf, BoundingBox &meshBBOx, int width,
              int height) {
  // From class s = delta yi / delta ym , ym being from bounding box
  float scaleX = static_cast<float>(width) / (meshBBOx.maxx - meshBBOx.minx);
  float scaleY = static_cast<float>(height) / (meshBBOx.maxy - meshBBOx.miny);

  // Use this
  float scale = min(scaleX, scaleY); // take smaller one

  // Calculate Translation as seen in class
  float middleOfImageX = width / 2.0f;
  float middleOfImageY = height / 2.0f;
  float middleOfMeshBBoxX = 0.5f * (meshBBOx.minx + meshBBOx.maxx);
  float middleOfMeshBBoxY = 0.5f * (meshBBOx.miny + meshBBOx.maxy);

  float translationX = middleOfImageX - (scale * middleOfMeshBBoxX);
  float translationY = middleOfImageY - (scale * middleOfMeshBBoxY);

  // Transform
  for (size_t i = 0; i < posBuf.size(); i += 3) {
    float newX = (scale * posBuf[i]) + translationX;
    float newY = (scale * posBuf[i + 1]) + translationY;
    posBuf[i] = newX;
    posBuf[i + 1] = newY;
  }
}
