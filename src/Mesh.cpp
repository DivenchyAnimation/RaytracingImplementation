#include "Mesh.h"

using std::vector, std::shared_ptr, std::make_shared, std::string, std::sqrt, glm::vec3, glm::vec4, glm::mat3, glm::mat4;

shared_ptr<Hit> Mesh::computeIntersection(const Ray &ray, const glm::mat4 modelMat, const glm::mat4 modelMatInv,
                                          const std::vector<Light> &lights) {
  shared_ptr<Hit> hit = make_shared<Hit>(); // assume collision is false

  // Help from ChatGPT making this model to object space transformation
  // Transform ray origin into local space (use homogeneous coordinate 1)
  glm::vec3 rayLocalOrigin = glm::vec3(modelMatInv * glm::vec4(ray.rayOrigin, 1.0f));

  // Transform ray direction into local space (use homogeneous coordinate 0)
  glm::vec3 rayLocalDirection = glm::normalize(glm::vec3(modelMatInv * glm::vec4(ray.rayDirection, 0.0f)));
  Ray localRay(rayLocalOrigin, rayLocalDirection);

  // Ray intersection test
  float t_bounding;
  if (!meshBoundingSphere.intersect(localRay, 0.001f, std::numeric_limits<float>::max(), t_bounding)) {
    return hit; // Misses the bunny
  }

  // Help from ChatGPT
  // Loop over the triangles
  float T = std::numeric_limits<float>::infinity();
  float U = 0, V = 0;
  size_t bestTri = size_t(-1);
  for (size_t i = 0; i < posBuf.size(); i += 9) {
    double orig[3] = {double(localRay.rayOrigin.x), double(localRay.rayOrigin.y), double(localRay.rayOrigin.z)};
    double dir[3] = {double(localRay.rayDirection.x), double(localRay.rayDirection.y), double(localRay.rayDirection.z)};
    double v0[3] = {posBuf[i + 0], posBuf[i + 1], posBuf[i + 2]};
    double v1[3] = {posBuf[i + 3], posBuf[i + 4], posBuf[i + 5]};
    double v2[3] = {posBuf[i + 6], posBuf[i + 7], posBuf[i + 8]};
    double t, u, v;
    // Use the optimizer codes for triange ray intersections
    if (intersect_triangle(orig, dir, v0, v1, v2, &t, &u, &v) && t > 0 && t < T) {
      T = float(t);
      U = float(u);
      V = float(v);
      bestTri = i / 9;
    }
  }

  // If after looping no triangle was hit
  if (bestTri == size_t(-1)) {
    return hit;
  }

  // Hit found
  vec3 P = localRay.rayOrigin + localRay.rayDirection * T; // local
  size_t idx = bestTri * 9;
  // Interpolate normal (local)
  glm::vec3 n0(norBuf[idx + 0], norBuf[idx + 1], norBuf[idx + 2]), n1(norBuf[idx + 3], norBuf[idx + 4], norBuf[idx + 5]),
      n2(norBuf[idx + 6], norBuf[idx + 7], norBuf[idx + 8]);
  vec3 N = glm::normalize(n0 * (1 - U - V) + n1 * U + n2 * V); // local
  // now in world space
  vec4 P_w = modelMat * vec4(P, 1.0f);
  P = vec3(P_w / P_w.w);                                     // world
  N = glm::normalize(glm::transpose(mat3(modelMatInv)) * N); // world

  hit->collision = true;
  hit->t = glm::length(P - ray.rayOrigin);
  hit->x = P;
  hit->n = N;
  return hit;
};
