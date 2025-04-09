#include "Routines.h"

using std::vector, std::make_shared, std::string, std::max, std::sqrt;

// Helpers
void rotateVertices(vector<float> &pointBuf) {
  // Theta = pi /4
  float theta = M_PI / 4.0f;
  float cosTheta = cos(theta);
  float sinTheta = sin(theta);

  // Little help from Deepseek
  float rotMatrix[3][3] = {
      {cosTheta, 0, sinTheta}, {0, 1, 0}, {-1 * sinTheta, 0, cosTheta}};

  // Calculate rotation
  // posBuf[i] --> x
  // posBuf[i+1] --> y
  // posBuf[i+2] --> z
  // Using easy method from class
  for (size_t i = 0; i < pointBuf.size(); i += 3) {
    float x = pointBuf[i] * rotMatrix[0][0] +
              pointBuf[i + 1] * rotMatrix[0][1] +
              pointBuf[i + 2] * rotMatrix[0][2];
    float y = pointBuf[i] * rotMatrix[1][0] +
              pointBuf[i + 1] * rotMatrix[1][1] +
              pointBuf[i + 2] * rotMatrix[1][2];
    float z = pointBuf[i] * rotMatrix[2][0] +
              pointBuf[i + 1] * rotMatrix[2][1] +
              pointBuf[i + 2] * rotMatrix[2][2];
    pointBuf[i] = x;
    pointBuf[i + 1] = y;
    pointBuf[i + 2] = z;
  }
};

// Tasks
int taskOne(vector<float> &posBuf, vector<float> &norBuf, vector<float> &zBuf,
            int width, int height, string outfile) {
  // Convert 3D mesh to 2D image coords
  int task = 1;
  BoundingBox meshBBox = meshBoundBox(posBuf);
  meshTo2D(posBuf, meshBBox, width, height);

  // Make triangles using new coords
  vector<Triangle> triangles;
  makeTriangles(triangles, posBuf, norBuf, task);

  // Create the image. We're using a `shared_ptr`, a C++11 feature.
  auto image = make_shared<Image>(width, height);

  // Draw bounding box with pattern
  for (Triangle tri : triangles) {
    for (int y = tri.bbox.miny; y < tri.bbox.maxy; y++) {
      for (int x = tri.bbox.minx; x < tri.bbox.maxx; x++) {
        image->setPixel(x, y, tri.r, tri.g, tri.b);
      }
    }
  }

  // Write image to file
  image->writeToFile(outfile);

  // Happy exit
  return 0;
}

int taskTwo(vector<float> &posBuf, vector<float> &norBuf, vector<float> &zBuf,
            int width, int height, string outfile) {

  int task = 2;
  // Convert 3D mesh to 2D image coords
  BoundingBox meshBBox = meshBoundBox(posBuf);
  meshTo2D(posBuf, meshBBox, width, height);

  // Make triangles using new coords
  vector<Triangle> triangles;
  makeTriangles(triangles, posBuf, norBuf, task);

  // Create the image. We're using a `shared_ptr`, a C++11 feature.
  auto image = make_shared<Image>(width, height);

  // Draw bounding box with pattern
  for (Triangle tri : triangles) {
    for (int y = tri.bbox.miny; y < tri.bbox.maxy; y++) {
      for (int x = tri.bbox.minx; x < tri.bbox.maxx; x++) {
        float alpha, beta, gamma;
        computeBarycentricCoords(tri.v1, tri.v2, tri.v3, x, y, alpha, beta,
                                 gamma);
        // Is point inside triangle?
        if (alpha >= -epsilon && beta >= -epsilon && gamma >= -epsilon) {
          image->setPixel(x, y, tri.r, tri.g, tri.b);
        }
      }
    }
  }

  // Write image to file
  image->writeToFile(outfile);

  return 0;
}

// Interpolate colors with barycentric coords
int taskThree(vector<float> &posBuf, vector<float> &norBuf, vector<float> &zBuf,
              int width, int height, string outfile) {

  // Convert 3D mesh to 2D image coords
  BoundingBox meshBBox = meshBoundBox(posBuf);
  meshTo2D(posBuf, meshBBox, width, height);

  // Make triangles using new coords
  int task = 3;
  vector<Triangle> triangles;
  makeTriangles(triangles, posBuf, norBuf, task);

  // Create the image. We're using a `shared_ptr`, a C++11 feature.
  auto image = make_shared<Image>(width, height);

  // Draw bounding box with pattern
  for (Triangle tri : triangles) {
    for (int y = tri.bbox.miny; y < tri.bbox.maxy; y++) {
      for (int x = tri.bbox.minx; x < tri.bbox.maxx; x++) {
        float alpha, beta, gamma;
        computeBarycentricCoords(tri.v1, tri.v2, tri.v3, x, y, alpha, beta,
                                 gamma);
        // Is point inside triangle?
        if (alpha >= -epsilon && beta >= -epsilon && gamma >= -epsilon) {
          // Interpolate color using barycentric cords
          unsigned char r = static_cast<unsigned char>(
              alpha * tri.v1.r + beta * tri.v2.r + gamma * tri.v3.r);
          unsigned char g = static_cast<unsigned char>(
              alpha * tri.v1.g + beta * tri.v2.g + gamma * tri.v3.g);
          unsigned char b = static_cast<unsigned char>(
              alpha * tri.v1.b + beta * tri.v2.b + gamma * tri.v3.b);

          image->setPixel(x, y, r, g, b);
        }
      }
    }
  }

  // Write image to file
  image->writeToFile(outfile);

  return 0;
}

// Linear interpolate from red and blue
int taskFour(vector<float> &posBuf, vector<float> &norBuf, vector<float> &zBuf,
             int width, int height, string outfile) {
  // Convert 3D mesh to 2D image coords
  int task = 4;
  BoundingBox meshBBox = meshBoundBox(posBuf);
  meshTo2D(posBuf, meshBBox, width, height);

  // Make triangles using new coords
  vector<Triangle> triangles;
  makeTriangles(triangles, posBuf, norBuf, task);

  BoundingBox gradientBox = meshBoundBox(posBuf);

  // Create the image. We're using a `shared_ptr`, a C++11 feature.
  auto image = make_shared<Image>(width, height);

  // Draw bounding box with pattern
  for (Triangle tri : triangles) {
    for (int y = tri.bbox.miny; y < tri.bbox.maxy; y++) {
      for (int x = tri.bbox.minx; x < tri.bbox.maxx; x++) {
        float alpha, beta, gamma;
        computeBarycentricCoords(tri.v1, tri.v2, tri.v3, x, y, alpha, beta,
                                 gamma);
        // Is point inside triangle?
        if (alpha >= -epsilon && beta >= -epsilon && gamma >= -epsilon) {
          // Help with ChatGPT
          float total =
              (y - gradientBox.miny) / (gradientBox.maxy - gradientBox.miny);
          image->setPixel(x, y, (total * 255), 0, ((1 - total) * 255));
        }
      }
    }
  }

  // Write image to file
  image->writeToFile(outfile);

  return 0;
}

// Z buffer image translation with the aid of ChatGPT
int taskFive(vector<float> &posBuf, vector<float> &norBuf, vector<float> &zBuf,
             int width, int height, string outfile) {
  // Convert 3D mesh to 2D image coords
  int task = 5;
  BoundingBox meshBBox = meshBoundBox(posBuf);
  meshTo2D(posBuf, meshBBox, width, height);

  // Init zBuf to -inf, ChatGPT
  zBuf.assign(width * height, -std::numeric_limits<float>::infinity());

  // Make triangles using new coords
  vector<Triangle> triangles;
  makeTriangles(triangles, posBuf, norBuf, task);

  // Create the image. We're using a `shared_ptr`, a C++11 feature.
  auto image = make_shared<Image>(width, height);

  // Draw bounding box with pattern
  for (Triangle tri : triangles) {
    for (int y = tri.bbox.miny; y < tri.bbox.maxy; y++) {
      for (int x = tri.bbox.minx; x < tri.bbox.maxx; x++) {
        float alpha, beta, gamma;
        computeBarycentricCoords(tri.v1, tri.v2, tri.v3, x, y, alpha, beta,
                                 gamma);
        // Is point inside triangle?
        if (alpha >= -epsilon && beta >= -epsilon && gamma >= -epsilon) {
          float z = alpha * tri.v1.z + beta * tri.v2.z + gamma * tri.v3.z;
          int index = y * width + x; // Provided by ChatGPT

          // Z test check for distance from camera, if closer redraw pixel
          if (z > zBuf[index]) {
            zBuf[index] = z;
            float normalizedZ =
                (z - meshBBox.minz) /
                (meshBBox.maxz - meshBBox.minz); // Provided by ChatGPT
            int redValue = static_cast<int>(normalizedZ * 255);

            image->setPixel(x, y, redValue, 0, 0);
          }
        }
      }
    }
  }
  // Write image to file
  image->writeToFile(outfile);

  return 0;
}

int taskSix(vector<float> &posBuf, vector<float> &norBuf, vector<float> &zBuf,
            int width, int height, string outfile) {
  // Convert 3D mesh to 2D image coords
  int task = 6;
  BoundingBox meshBBox = meshBoundBox(posBuf);
  meshTo2D(posBuf, meshBBox, width, height);

  // Init zBuf to -inf, ChatGPT
  zBuf.assign(width * height, -std::numeric_limits<float>::infinity());

  // Make triangles using new coords
  vector<Triangle> triangles;
  makeTriangles(triangles, posBuf, norBuf, task);
  // BoundingBoxZ zBBox = meshBoundBoxZ(posBuf);

  // Create the image. We're using a `shared_ptr`, a C++11 feature.
  auto image = make_shared<Image>(width, height);

  // Draw bounding box with pattern
  for (Triangle tri : triangles) {
    for (int y = tri.bbox.miny; y < tri.bbox.maxy; y++) {
      for (int x = tri.bbox.minx; x < tri.bbox.maxx; x++) {
        float alpha, beta, gamma;
        computeBarycentricCoords(tri.v1, tri.v2, tri.v3, x, y, alpha, beta,
                                 gamma);
        // Is point inside triangle?
        if (alpha >= -epsilon && beta >= -epsilon && gamma >= -epsilon) {
          float z = alpha * tri.v1.z + beta * tri.v2.z + gamma * tri.v3.z;
          int index = y * width + x; // Provided by ChatGPT

          // Z test check for distance from camera, if closer redraw pixel
          if (z > zBuf[index]) {
            zBuf[index] = z;
            float _x = tri.v1.nx * alpha + tri.v2.nx * beta + tri.v3.nx * gamma;
            float _y = tri.v1.ny * alpha + tri.v2.ny * beta + tri.v3.ny * gamma;
            float _z = tri.v1.nz * alpha + tri.v2.nz * beta + tri.v3.nz * gamma;

            int _r = static_cast<int>(255 * (0.5 * _x + 0.5));
            int _g = static_cast<int>(255 * (0.5 * _y + 0.5));
            int _b = static_cast<int>(255 * (0.5 * _z + 0.5));

            image->setPixel(x, y, _r, _g, _b);
          }
        }
      }
    }
  }
  // Write image to file
  image->writeToFile(outfile);

  return 0;
}

int taskSeven(vector<float> &posBuf, vector<float> &norBuf, vector<float> &zBuf,
              int width, int height, string outfile) {
  // Convert 3D mesh to 2D image coords
  int task = 7;
  BoundingBox meshBBox = meshBoundBox(posBuf);
  meshTo2D(posBuf, meshBBox, width, height);

  // Init zBuf to -inf, ChatGPT
  zBuf.assign(width * height, -std::numeric_limits<float>::infinity());

  // Make triangles using new coords
  vector<Triangle> triangles;
  makeTriangles(triangles, posBuf, norBuf, task);
  // BoundingBoxZ zBBox = meshBoundBoxZ(posBuf);

  // Create the image. We're using a `shared_ptr`, a C++11 feature.
  auto image = make_shared<Image>(width, height);

  // Draw bounding box with pattern
  for (Triangle tri : triangles) {
    for (int y = tri.bbox.miny; y < tri.bbox.maxy; y++) {
      for (int x = tri.bbox.minx; x < tri.bbox.maxx; x++) {
        float alpha, beta, gamma;
        computeBarycentricCoords(tri.v1, tri.v2, tri.v3, x, y, alpha, beta,
                                 gamma);
        // Is point inside triangle?
        if (alpha >= -epsilon && beta >= -epsilon && gamma >= -epsilon) {
          float z = alpha * tri.v1.z + beta * tri.v2.z + gamma * tri.v3.z;
          int index = y * width + x; // Provided by ChatGPT

          // Z test check for distance from camera, if closer redraw pixel
          if (z > zBuf[index]) {
            zBuf[index] = z;
            // Using formulas in the assignment
            float _x = tri.v1.nx * alpha + tri.v2.nx * beta + tri.v3.nx * gamma;
            float _y = tri.v1.ny * alpha + tri.v2.ny * beta + tri.v3.ny * gamma;
            float _z = tri.v1.nz * alpha + tri.v2.nz * beta + tri.v3.nz * gamma;

            vector<float> light = {static_cast<float>(1 / sqrt(3)),
                                   static_cast<float>(1 / sqrt(3)),
                                   static_cast<float>(1 / sqrt(3))};
            float lightNormalDot =
                (light[0] * _x) + (light[1] * _y) + (light[2] * _z);
            float c = max(lightNormalDot, 0.0f);

            image->setPixel(x, y, c * 255, c * 255, c * 255);
          }
        }
      }
    }
  }
  // Write image to file
  image->writeToFile(outfile);

  return 0;
}

int taskEight(vector<float> &posBuf, vector<float> &norBuf, vector<float> &zBuf,
              int width, int height, string outfile) {
  // Rotate mesh before any other operations
  rotateVertices(posBuf);
  // Don't forget normals!!!
  rotateVertices(norBuf);

  // Convert 3D mesh to 2D image coords
  int task = 8;
  BoundingBox meshBBox = meshBoundBox(posBuf);
  meshTo2D(posBuf, meshBBox, width, height);

  // Init zBuf to -inf, ChatGPT
  zBuf.assign(width * height, -std::numeric_limits<float>::infinity());

  // Make triangles using new coords
  vector<Triangle> triangles;
  makeTriangles(triangles, posBuf, norBuf, task);
  // BoundingBoxZ zBBox = meshBoundBoxZ(posBuf);

  // Create the image. We're using a `shared_ptr`, a C++11 feature.
  auto image = make_shared<Image>(width, height);

  // Draw bounding box with pattern
  for (Triangle tri : triangles) {
    for (int y = tri.bbox.miny; y < tri.bbox.maxy; y++) {
      for (int x = tri.bbox.minx; x < tri.bbox.maxx; x++) {
        float alpha, beta, gamma;
        computeBarycentricCoords(tri.v1, tri.v2, tri.v3, x, y, alpha, beta,
                                 gamma);
        // Is point inside triangle?
        if (alpha >= -epsilon && beta >= -epsilon && gamma >= -epsilon) {
          float z = alpha * tri.v1.z + beta * tri.v2.z + gamma * tri.v3.z;
          int index = y * width + x; // Provided by ChatGPT

          // Z test check for distance from camera, if closer redraw pixel
          if (z > zBuf[index]) {
            zBuf[index] = z;
            // Using formulas in the assignment
            float _x = tri.v1.nx * alpha + tri.v2.nx * beta + tri.v3.nx * gamma;
            float _y = tri.v1.ny * alpha + tri.v2.ny * beta + tri.v3.ny * gamma;
            float _z = tri.v1.nz * alpha + tri.v2.nz * beta + tri.v3.nz * gamma;

            vector<float> light = {static_cast<float>(1 / sqrt(3)),
                                   static_cast<float>(1 / sqrt(3)),
                                   static_cast<float>(1 / sqrt(3))};
            float lightNormalDot =
                (light[0] * _x) + (light[1] * _y) + (light[2] * _z);
            float c = max(lightNormalDot, 0.0f);

            image->setPixel(x, y, c * 255, c * 255, c * 255);
          }
        }
      }
    }
  }
  // Write image to file
  image->writeToFile(outfile);

  return 0;
}
