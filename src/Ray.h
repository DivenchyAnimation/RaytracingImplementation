#pragma once

struct Ray {
  glm::vec3 rayOrigin;
  glm::vec3 rayDirection;
  Ray(glm::vec3 rayOrigin, glm::vec3 rayDirection) : rayOrigin(rayOrigin), rayDirection(rayDirection) {}
};
