#pragma once
enum class ShapeType { SPHERE, ELLIPSOID, CUBE, CYLINDER, PLANE };

class Shape {
protected:
  ShapeType type;
  glm::vec3 position; // Center of shape
  //glm::vec3 normal;   // For planes
  glm::vec3 rotation;
  glm::vec3 scale;
  //float radius;
  //float height;
  //float width;
  float rotationAngle;
  std::shared_ptr<Material> material;

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
  float getRotationAngle() { return this->rotationAngle; };
  std::shared_ptr<Material> getMaterial() { return this->material; };
  ShapeType getType() { return this->type; };
  glm::vec3 getScale() { return this->scale; };
  glm::vec3 getRotationAxis() { return this->rotation; };
  void setRotationAxis(glm::vec3 rotVec) { this->rotation = rotVec; };
  void setRotationAngle(float angle) { this->rotationAngle = angle; };
};
