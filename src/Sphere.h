#pragma once

class Sphere : public Shape {

private:
  float radius;

public:
  Sphere(glm::vec3 position, float radius, float scale, float rotAngle, std::shared_ptr<Material> material) : Shape() {
    this->type = ShapeType::SPHERE;
    this->position = position;
    this->material = material;
    this->rotationAngle = rotAngle;
    this->radius = radius;
    this->scale = glm::vec3(scale);
  };

  float getRadius() { return this->radius; };
  virtual std::shared_ptr<Hit> computeIntersection(const Ray &ray, const glm::mat4 modelMat, const glm::mat4 modelMatInv,
                                                   const std::vector<Light> &lights) override;
};
