#include "pch.h"

enum class ShapeType { SPHERE, ELLIPSOID, CUBE, CYLINDER };

class Shape {
private:
  ShapeType type;
  glm::vec3 position; // Center of shape
  glm::vec3 scale;
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
  // For easy sphere
  Shape(ShapeType type, glm::vec3 position, float radius, std::shared_ptr<Material> material) {
    this->type = type;
    this->position = position;
    this->radius = radius;
    this->material = material;
    this->height = 0.0f;
    this->width = 0.0f;
  }
  // For other shapes
  Shape(ShapeType type, glm::vec3 position, float radius, float height, float width, std::shared_ptr<Material> material) {
    this->type = type;
    this->position = position;
    this->radius = radius;
    this->height = height;
    this->width = width;
    this->material = material;
  }
  ~Shape() {};
  glm::vec3 getPosition() { return this->position; };
  float getRadius() { return this->radius; };
  std::shared_ptr<Material> getMaterial() { return this->material; };
};
