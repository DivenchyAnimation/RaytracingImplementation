// src/ Routines.cpp
#include "Routines.h"
#include "pch.h"
#include <memory>

using std::vector, std::shared_ptr, std::make_shared, std::string, std::sqrt, glm::vec3;

// Helpers
void rotateVertices(vector<float> &pointBuf) {
  // Theta = pi /4
  float theta = M_PI / 4.0f;
  float cosTheta = cos(theta);
  float sinTheta = sin(theta);

  // Little help from Deepseek
  float rotMatrix[3][3] = {
      {cosTheta,      0, sinTheta},
      {0,             1, 0       },
      {-1 * sinTheta, 0, cosTheta}
  };

  // Calculate rotation
  // posBuf[i] --> x
  // posBuf[i+1] --> y
  // posBuf[i+2] --> z
  // Using easy method from class
  for (size_t i = 0; i < pointBuf.size(); i += 3) {
    float x = pointBuf[i] * rotMatrix[0][0] + pointBuf[i + 1] * rotMatrix[0][1] + pointBuf[i + 2] * rotMatrix[0][2];
    float y = pointBuf[i] * rotMatrix[1][0] + pointBuf[i + 1] * rotMatrix[1][1] + pointBuf[i + 2] * rotMatrix[1][2];
    float z = pointBuf[i] * rotMatrix[2][0] + pointBuf[i + 1] * rotMatrix[2][1] + pointBuf[i + 2] * rotMatrix[2][2];
    pointBuf[i] = x;
    pointBuf[i + 1] = y;
    pointBuf[i + 2] = z;
  }
};

// Helper function
float clamp(float x, float minVal = 0.0f, float maxVal = 1.0f) { return std::max(minVal, std::min(x, maxVal)); }

void initMaterials(std::vector<std::shared_ptr<Material>> &materials) {
  shared_ptr<Material> redMaterial =
      make_shared<Material>(vec3(0.1f, 0.1f, 0.1f), vec3(1.0f, 0.0f, 0.0f), vec3(1.0f, 1.0f, 0.5f), 100.0f);
  shared_ptr<Material> greenMaterial =
      make_shared<Material>(vec3(0.1f, 0.1f, 0.1f), vec3(0.0f, 1.0f, 0.0f), vec3(1.0f, 1.0f, 0.5f), 100.0f);
  shared_ptr<Material> blueMaterial =
      make_shared<Material>(vec3(0.1f, 0.1f, 0.1f), vec3(0.0f, 0.0f, 1.0f), vec3(1.0f, 1.0f, 0.5f), 100.0f);
  materials.push_back(redMaterial);
  materials.push_back(greenMaterial);
  materials.push_back(blueMaterial);
}

vec3 genRayForPixel(int x, int y, int width, int height, float fov) {
  float planeDistFromCam = 1.0f;
  float halfWidth = planeDistFromCam * tan((fov / 2.0f));
  float halfHeight = halfWidth;

  // Help from ChatGPT
  // Map pixel (x,y) to norm coords
  float u = ((x + 0.5f) / width) * 2.0f - 1.0f;
  float v = ((y + 0.5f) / height) * 2.0f - 1.0f;
  return glm::normalize(vec3(u * halfWidth, v * halfHeight, -planeDistFromCam));
}

// Help from ChatGPT
// Compute intersections of ray with shape in order to create the shading
shared_ptr<Hit> computeIntersection(const Ray &ray, const shared_ptr<Shape> &shape, const Light &light) {
  shared_ptr<Hit> hit = make_shared<Hit>(); // assume collision is false
  // Compute quadratic equation
  vec3 oc = ray.rayOrigin - shape->getPosition();
  float a = glm::dot(ray.rayDirection, ray.rayDirection);
  float b = 2.0f * glm::dot(oc, ray.rayDirection);
  float c = glm::dot(oc, oc) - shape->getRadius() * shape->getRadius();
  float discriminant = b * b - 4 * a * c;

  // If discriminant is negative, no collision with shape
  if (discriminant < 0) {
    return hit;
  }

  // Else, colllision so solve for t (intersections)
  float sqrtDiscriminant = sqrt(discriminant);
  float t = (-b - sqrtDiscriminant) / (2.0f * a);

  // Behind camera, count as miss look at other t val
  if (t < 0) {
    t = (-b + sqrtDiscriminant) / (2.0f * a);
  }
  // If both neg, skip
  if (t < 0) {
    return hit;
  }

  // Ray hit
  hit->t = t;
  vec3 P = ray.rayOrigin + t * ray.rayDirection;     // Intersection point
  vec3 N = glm::normalize(P - shape->getPosition()); // Normal at intersection point
  vec3 L = glm::normalize(light.pos - P);            // Light direction
  hit->x = P;
  hit->n = N;

  // Diffuse shading
  // Diffuse shading (Lambertian).
  float diff = std::max(glm::dot(N, L), 0.0f);
  glm::vec3 diffuse = light.intensity * shape->getMaterial()->getMaterialKD() * diff;
  // Specular shading (Phong reflection).
  glm::vec3 V = glm::normalize(ray.rayOrigin - P); // View direction.
  glm::vec3 R = glm::reflect(-L, N);
  float spec = pow(std::max(glm::dot(V, R), 0.0f), shape->getMaterial()->getMaterialS());
  glm::vec3 specular = light.intensity * shape->getMaterial()->getMaterialKS() * spec;
  // Combine with ambient lighting.
  glm::vec3 color = shape->getMaterial()->getMaterialKA() + diffuse + specular;

  // Clamp the color components.
  color.r = clamp(color.r);
  color.g = clamp(color.g);
  color.b = clamp(color.b);
  hit->color = color;
  hit->collision = true;

  return hit;
}

bool isInShadow(shared_ptr<Hit> nearestHit, const Light &light, const vector<shared_ptr<Shape>> &shapes) {

  // Shadow test
  float epsilon = 0.001f;
  vec3 shadowOrigin = nearestHit->x + nearestHit->n * epsilon;
  Ray shadowRay(shadowOrigin, glm::normalize(light.pos - shadowOrigin));
  float lightDistance = glm::length(light.pos - shadowOrigin);
  for (shared_ptr<Shape> shape : shapes) {
    if (nearestHit->collisionShape == shape) {
      continue; // Don't double count shape that caused initial hit
    }
    shared_ptr<Hit> shadowHit = computeIntersection(shadowRay, shape, light);
    if ((shadowHit->collision) && (shadowHit->t > 0) && (shadowHit->t < lightDistance)) {
      return true;
    }
  }
  return false;
}

void genScenePixels(Image &image, int width, int height, shared_ptr<Camera> &camPos, const vector<shared_ptr<Shape>> &shapes,
                    vector<shared_ptr<Hit>> &hits, const Light &light, string FILENAME) {
  shared_ptr<Hit> nearestHit = make_shared<Hit>();
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      nearestHit = nullptr;
      vec3 rayDir = genRayForPixel(x, y, width, height, camPos->getFOVY());

      // Initalize a dummy tVal
      float nearestToCamT = std::numeric_limits<float>::max();
      vec3 finalColor(0.0f, 0.0f, 0.0f); // black bachground

      // Check intersections on each shape
      for (shared_ptr<Shape> shape : shapes) {
        shared_ptr<Hit> curHit = computeIntersection(Ray(camPos->getPosition(), rayDir), shape, light);
        if (curHit->collision && curHit->t < nearestToCamT) {
          nearestToCamT = curHit->t;
          finalColor = curHit->color;
          curHit->collisionShape = shape;
          nearestHit = curHit;
        }
      }

      // If hit exists, do shadow test
      if ((nearestHit != nullptr) && isInShadow(nearestHit, light, shapes)) {
        // If in shadow, set color to black
        finalColor = vec3(0.1f, 0.1f, 0.1f);
      }

      // Convert color components from [0,1] to [0,255].
      unsigned char rVal = static_cast<unsigned char>(finalColor.r * 255);
      unsigned char gVal = static_cast<unsigned char>(finalColor.g * 255);
      unsigned char bVal = static_cast<unsigned char>(finalColor.b * 255);

      // Set the pixel color in the image.
      image.setPixel(x, y, rVal, gVal, bVal);
    }
  }

  // Write to image
  image.writeToFile(FILENAME);
  std::cout << "Image written to " << FILENAME << std::endl;
}

void taskOne(int width, int height, std::vector<std::shared_ptr<Material>> materials, std::vector<std::shared_ptr<Shape>> &shapes,
             std::vector<std::shared_ptr<Hit>> &hits, std::shared_ptr<Camera> &cam, std::string FILENAME) {
  Image image(width, height);
  // Shape(ShapeType type, glm::vec3 position, float radius, std::shared_ptr<Material> material)
  shared_ptr<Shape> redSphere = make_shared<Shape>(ShapeType::SPHERE, vec3(-0.5f, -1.0f, 1.0f), 1.0f, materials[0]);
  shared_ptr<Shape> greenSphere = make_shared<Shape>(ShapeType::SPHERE, vec3(0.5f, -1.0f, -1.0f), 1.0f, materials[1]);
  shared_ptr<Shape> blueSphere = make_shared<Shape>(ShapeType::SPHERE, vec3(0.0f, 1.0f, 0.0f), 1.0f, materials[2]);
  shapes.push_back(redSphere);
  shapes.push_back(greenSphere);
  shapes.push_back(blueSphere);
  Light worldLight(vec3(-2.0f, 1.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f), 1.0f);

  genScenePixels(image, width, height, cam, shapes, hits, worldLight, FILENAME);
}
