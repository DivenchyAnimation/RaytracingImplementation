#pragma once

class Light {
  // Make it public so don't have to bother with getters
public:
  glm::vec3 pos;
  glm::vec3 color;
  float baseAngle;
  float ringRadius;
  float intensity;

  Light(const glm::vec3 &pos, const glm::vec3 &color, float intensity) {
    this->pos = pos;
    this->color = color;
    this->intensity = intensity;
  };

  Light(const glm::vec3 &pos, const glm::vec3 &color, const float baseAngle, const float ringRadius) {
    this->pos = pos;
    this->color = color;
    this->baseAngle = baseAngle;
    this->ringRadius = ringRadius;
    this->intensity = 1.0f;
  };
};
