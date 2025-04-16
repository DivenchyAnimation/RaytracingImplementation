#pragma once
class Ellipsoid : public Shape {
private:
  float radius;

public:
  Ellipsoid() : Shape() {
    this->type = ShapeType::ELLIPSOID;
    this->radius = 1.0f;
  }
  Ellipsoid(glm::vec3 position, float radius, glm::vec3 scale, float rotAngle, std::shared_ptr<Material> material) : Shape() {
    this->type = ShapeType::ELLIPSOID;
    this->position = position;
    this->material = material;
    this->rotationAngle = rotAngle;
    this->radius = radius;
    this->scale = scale;
  }
  ~Ellipsoid() override = default;

  float getRadius() { return this->radius; };
  virtual std::shared_ptr<Hit> computeIntersection(const Ray &ray, const glm::mat4 modelMat, const glm::mat4 modelMatInv,
                                                   const std::vector<Light> &lights) override;
};
