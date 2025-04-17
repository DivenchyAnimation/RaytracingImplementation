#pragma once
enum class ShapeType { SPHERE, ELLIPSOID, CUBE, CYLINDER, PLANE };

class Shape {
protected:
  ShapeType type = ShapeType::SPHERE;               // Default shape type
  glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f); // Center of shape
  glm::vec3 rotation = glm::vec3(0.0f, 1.0f, 0.0f); // Axis of rotation
  glm::vec3 scale = glm::vec3(1.0f);
  float rotationAngle = 0.0f;
  std::shared_ptr<Material> material;
  bool isReflective = false; // Default is not reflective

public:
  // Default Shape
  Shape() {
    this->type = ShapeType::SPHERE;
    this->material = std::make_shared<Material>();
    this->position = glm::vec3(0.0f, 0.0f, 0.0f); // Place in world origin by default
    this->scale = glm::vec3(1.0f, 1.0f, 1.0f);    // Default scale
  }

  virtual ~Shape() = default;
  virtual std::shared_ptr<Hit> computeIntersection(const Ray &ray, const glm::mat4 modelMat, const glm::mat4 modelMatInv,
                                                   const std::vector<Light> &lights) = 0;
  glm::vec3 getPosition() { return this->position; };
  bool getIsReflective() { return this->isReflective; };
  float getRotationAngle() { return this->rotationAngle; };
  std::shared_ptr<Material> getMaterial() { return this->material; };
  ShapeType getType() { return this->type; };
  glm::vec3 getScale() { return this->scale; };
  glm::vec3 getRotationAxis() { return this->rotation; };
  void setRotationAxis(glm::vec3 rotVec) { this->rotation = rotVec; };
  void setRotationAngle(float angle) { this->rotationAngle = angle; };
};
