#pragma once

class Shape;
class Hit {
public:
  Hit() : x(0), n(0), t(0) {}
  Hit(const glm::vec3 &x, const glm::vec3 &n, float t) {
    this->x = x;
    this->n = n;
    this->t = t;
  }
  glm::vec3 x;     // position
  glm::vec3 n;     // normal
  float t;         // distance
  glm::vec3 color; // color at hit (shading applied)
  bool collision = false;
  std::shared_ptr<Shape> collisionShape; // shape that was hit
};
