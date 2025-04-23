#pragma once
#include <glm/glm.hpp>
#include <cuda_runtime.h>

struct GPUShape;
struct GPUHit {
  glm::vec3 x;     // position
  glm::vec3 n;     // normal
  float t;         // distance
  glm::vec3 color; // color at hit (shading applied)
  bool collision = false;
  GPUShape *collisionShape = NULL; // shape that was hit
};
