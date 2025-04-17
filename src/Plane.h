#pragma once

class Plane : public Shape {
private:
  glm::vec3 normal = glm::vec3(0.0f, 1.0f, 0.0f);

public:
  Plane(glm::vec3 position, glm::vec3 normal, float scale, float rotAngle, std::shared_ptr<Material> material) : Shape() {
    this->type = ShapeType::PLANE;
    this->position = position;
    this->normal = normal;
    this->material = material;
    this->rotationAngle = rotAngle;
    this->scale = glm::vec3(scale);
  };

  virtual std::shared_ptr<Hit> computeIntersection(const Ray &ray, const glm::mat4 modelMat, const glm::mat4 modelMatInv,
                                                   const std::vector<Light> &lights) override;
};
