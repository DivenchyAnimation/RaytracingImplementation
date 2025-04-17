
struct BoundingSphere {
public:
  glm::vec3 center;
  float radius;

  // Default
  BoundingSphere() : center(0.0f), radius(0.0f) {};
  BoundingSphere(std::vector<float> &posBuf);
  bool intersect(const Ray &ray, float t_min, float t_max, float &hitT) const;
};
