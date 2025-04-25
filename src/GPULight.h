#pragma once
#include "GPUVecOps.h"

struct GPULight {
  vec3 pos;
  vec3 color;
  float baseAngle;
  float ringRadius;
  float intensity;

  HD GPULight() {
      this->pos = vec3(0.0f);
      this->color = vec3(0.0f);
      this->intensity = 0.0f;
      this->baseAngle = 0.0f;
      this->ringRadius = 0.0f;
  }

  HD GPULight(const vec3 &pos, const vec3 &color, float intensity) {
    this->pos = pos;
    this->color = color;
    this->intensity = intensity;
    this->baseAngle = 0.0f;
    this->ringRadius = 0.0f;
  };

  HD GPULight(const vec3 &pos, const vec3 &color, const float baseAngle, const float ringRadius) {
    this->pos = pos;
    this->color = color;
    this->baseAngle = baseAngle;
    this->ringRadius = ringRadius;
    this->intensity = 1.0f;
  };
};
