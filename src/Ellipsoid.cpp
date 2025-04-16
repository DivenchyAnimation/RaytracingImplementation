#include "Ellipsoid.h"

using std::vector, std::shared_ptr, std::make_shared, std::string, std::sqrt, glm::vec3, glm::vec4, glm::mat4;

// Help from ChatGPT
// Compute intersections of ray with shape in order to create the shading
shared_ptr<Hit> Ellipsoid::computeIntersection(const Ray &ray, const mat4 modelMat, const mat4 modelMatInv,
                                               const vector<Light> &lights) {
  shared_ptr<Hit> hit = make_shared<Hit>(); // assume collision is false

  // Help from ChatGPT making this model to object space transformation

  // Transform ray origin into local space (use homogeneous coordinate 1)
  glm::vec3 localOrigin = glm::vec3(modelMatInv * glm::vec4(ray.rayOrigin, 1.0f));

  // Transform ray direction into local space (use homogeneous coordinate 0)
  glm::vec3 localDirection = glm::normalize(glm::vec3(modelMatInv * glm::vec4(ray.rayDirection, 0.0f)));

  // Compute quadratic equation
  float a, b, c, discriminant;
  vec3 oc = localOrigin;
  a = glm::dot(localDirection, localDirection);
  b = 2.0f * glm::dot(oc, localDirection);
  c = glm::dot(oc, oc) - getRadius() * getRadius();
  discriminant = b * b - 4 * a * c;

  // If discriminant is negative, no collision with shape
  if (discriminant < 0) {
    return hit;
  }

  // Else, colllision so solve for t (intersections)
  float sqrtDiscriminant = sqrt(discriminant);
  float t = (-b - sqrtDiscriminant) / (2.0f * a);

  // Behind camera, count as miss look at other t val
  if (t < 0) {
    t = (-b + sqrtDiscriminant) / (2.0f * a);
  }
  // If both neg, skip
  if (t < 0) {
    return hit;
  }

  // Object space ray hit
  vec3 localHitPoint = localOrigin + t * localDirection;
  vec3 localNormal = glm::normalize(localHitPoint); // For sphere

  // World space ray hit (dividie by w to enter eye space)
  vec4 worldHitPoint4 = modelMat * glm::vec4(localHitPoint, 1.0f);
  vec3 worldHitPoint = glm::vec3(worldHitPoint4) / worldHitPoint4.w;

  // Transform normal to world space
  glm::mat4 invTransModel = glm::transpose(modelMatInv);
  glm::vec4 worldNormal4 = invTransModel * glm::vec4(localNormal, 0.0f);
  glm::vec3 worldNormal = glm::normalize(glm::vec3(worldNormal4));

  // Calc t in world space
  float t_world = glm::length(worldHitPoint - ray.rayOrigin);
  if (glm::dot(ray.rayDirection, worldHitPoint - ray.rayOrigin) < 0.0f) {
    t_world = -t_world;
  }

  // Ray hit
  hit->t = t_world;
  vec3 P = worldHitPoint; // Intersection point
  vec3 N = worldNormal;   // Normal at intersection point
  hit->x = P;
  hit->n = N;

  // Add all lights contribution to the color
  // hit->color = calcLightContribution(lights, P, N, shape, ray);
  hit->collision = true;

  return hit;
}
