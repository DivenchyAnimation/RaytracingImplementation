#include "Plane.h"

using std::vector, std::shared_ptr, std::make_shared, std::string, glm::vec3, glm::vec4, glm::mat4;

// Help from ChatGPT
// Compute intersections of ray with shape in order to create the shading
shared_ptr<Hit> Plane::computeIntersection(const Ray &ray, const mat4 modelMat, const mat4 modelMatInv,
                                           const vector<Light> &lights) {
  shared_ptr<Hit> hit = make_shared<Hit>(); // assume collision is false

  // For an infinite plane, we assume the plane passes through shape->getPosition()
  vec3 planePos = getPosition();
  glm::vec3 worldNormal = glm::normalize(glm::mat3(glm::transpose(modelMatInv)) * normal);

  float denom = glm::dot(worldNormal, ray.rayDirection);
  // Avoid div by zero
  if (fabs(denom) > 1e-6) {
    // Compute t: distance along ray to intersection
    float t = glm::dot(planePos - ray.rayOrigin, worldNormal) / denom;
    if (t >= 0.0f) { // only count intersections in front of the ray
      vec3 P = ray.rayOrigin + t * ray.rayDirection;

      hit->t = t;
      hit->x = P;
      hit->n = worldNormal; // constant normal for an infinite plane
      hit->collision = true;
    }
  }
  return hit;
}
