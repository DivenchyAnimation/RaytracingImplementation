#include "pch.h"

enum class ShapeType { SPHERE, ELLIPSOID, CUBE, CYLINDER };

class Shape {
private:
  ShapeType type;
  glm::vec3 position; // Center of shape
  float radius;
  float height;
  float width;
  std::shared_ptr<Material> material;

public:
  // Default Shape
  Shape() {
    this->type = ShapeType::SPHERE;
    this->material = std::make_shared<Material>();
    this->radius = 1.0f;
    this->height = 0.0f;
    this->width = 0.0f;
    this->position = glm::vec3(0.0f, 0.0f, 0.0f); // Place in world origin by default
  }
  ~Shape() {};
  glm::vec3 getPosition() { return this->position; };
  float getRadius() { return this->radius; };
  std::shared_ptr<Material> getMaterial() { return this->material; };
};
