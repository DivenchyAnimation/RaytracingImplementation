#pragma once

struct GPUShape;
struct GPUHit {
  vec3 x;     // position
  vec3 n;     // normal
  float t;         // distance
  vec3 color; // color at hit (shading applied)
  bool collision = false;
  GPUShape *collisionShape = NULL; // shape that was hit
};
