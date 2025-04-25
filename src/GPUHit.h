#pragma once

struct GPUShape;
struct GPUHit {
  vec3 x;     // position
  vec3 n;     // normal
  float t;         // distance
  vec3 color; // color at hit (shading applied)
  bool collision = false;
  GPUShape *collisionShape = NULL; // shape that was hit

  HD GPUHit() {
	  this->x = vec3(0.0f);
	  this->n = vec3(0.0f);
	  this->t = 0.0f;
	  this->color = 0.0f;
  }
};
