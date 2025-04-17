
using std::vector, std::min, std::max, std::numeric_limits;

BoundingSphere::BoundingSphere(std::vector<float> &posBuf) {
  float minX = posBuf[0];
  float maxX = posBuf[0];
  float minY = posBuf[1];
  float maxY = posBuf[1];
  float minZ = posBuf[2];
  float maxZ = posBuf[2];

  for (size_t i = 0; i < posBuf.size(); i += 3) {
    minX = std::min(minX, posBuf[i]);
    maxX = std::max(maxX, posBuf[i]);
    minY = std::min(minY, posBuf[i + 1]);
    maxY = std::max(maxY, posBuf[i + 1]);
    minZ = std::min(minZ, posBuf[i + 2]);
    maxZ = std::max(maxZ, posBuf[i + 2]);
  }

  glm::vec3 minVec(minX, minY, minZ);
  glm::vec3 maxVec(maxX, maxY, maxZ);
  this->center = (minVec + maxVec) / 2.0f;
  this->radius = glm::length(maxVec - center);
}

bool BoundingSphere::intersect(const Ray &ray, float t_min, float t_max, float &hitT) const {
  glm::vec3 oc = ray.rayOrigin - center;
  float a = glm::dot(ray.rayDirection, ray.rayDirection);
  float b = 2.0f * glm::dot(oc, ray.rayDirection);
  float c = glm::dot(oc, oc) - radius * radius;
  float disc = b * b - 4 * a * c;
  if (disc < 0.0f)
    return false;
  float sq = std::sqrt(disc);

  // nearer root
  float t0 = (-b - sq) / (2.0f * a);
  if (t0 < t_min || t0 > t_max) {
    // try farther root
    float t1 = (-b + sq) / (2.0f * a);
    if (t1 < t_min || t1 > t_max)
      return false;
    hitT = t1;
    return true;
  }
  hitT = t0;
  return true;
}
