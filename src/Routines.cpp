// src/ Routines.cpp
#include "Routines.h"
#include "pch.h"
#include <memory>

using std::vector, std::shared_ptr, std::make_shared, std::string, std::sqrt, glm::vec3, glm::vec4, glm::mat4;

// Helper function
float clamp(float x, float minVal = 0.0f, float maxVal = 1.0f) { return std::max(minVal, std::min(x, maxVal)); }

vec3 calcLightContribution(const vector<Light> lights, vec3 P, vec3 N, shared_ptr<Shape> shape, Ray ray) {
  glm::vec3 color = shape->getMaterial()->getMaterialKA();
  for (const Light &light : lights) {
    // Directin of light
    vec3 L = glm::normalize(light.pos - P);

    // Diffuse shading
    float diff = std::max(glm::dot(N, L), 0.0f);
    glm::vec3 diffuse = light.intensity * shape->getMaterial()->getMaterialKD() * diff;

    // Specular shading (Phong reflection).
    glm::vec3 V = glm::normalize(ray.rayOrigin - P); // View direction.
    glm::vec3 R = glm::reflect(-L, N);
    float spec = pow(std::max(glm::dot(V, R), 0.0f), shape->getMaterial()->getMaterialS());
    glm::vec3 specular = light.intensity * shape->getMaterial()->getMaterialKS() * spec;

    // Add contribution to color
    color += specular + diffuse;
  }
  // Clamp the color components.
  color.r = clamp(color.r);
  color.g = clamp(color.g);
  color.b = clamp(color.b);
  return color;
}

void initMaterials(std::vector<std::shared_ptr<Material>> &materials) {
  shared_ptr<Material> redMaterial =
      make_shared<Material>(vec3(0.1f, 0.1f, 0.1f), vec3(1.0f, 0.0f, 0.0f), vec3(1.0f, 1.0f, 0.5f), 100.0f);
  shared_ptr<Material> greenMaterial =
      make_shared<Material>(vec3(0.1f, 0.1f, 0.1f), vec3(0.0f, 1.0f, 0.0f), vec3(1.0f, 1.0f, 0.5f), 100.0f);
  shared_ptr<Material> blueMaterial =
      make_shared<Material>(vec3(0.1f, 0.1f, 0.1f), vec3(0.0f, 0.0f, 1.0f), vec3(1.0f, 1.0f, 0.5f), 100.0f);
  shared_ptr<Material> planeMaterial = make_shared<Material>(vec3(0.1f), vec3(1.0f), vec3(0.0f), 0.0f);
  materials.push_back(redMaterial);
  materials.push_back(greenMaterial);
  materials.push_back(blueMaterial);
  materials.push_back(planeMaterial);
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
shared_ptr<Hit> computeIntersectionSphere(const Ray &ray, const shared_ptr<Shape> &shape, const mat4 modelMat,
                                          const mat4 modelMatInv, const vector<Light> &lights) {
  shared_ptr<Hit> hit = make_shared<Hit>(); // assume collision is false

  // Help from ChatGPT making this model to object space transformation

  // Transform ray origin into local space (use homogeneous coordinate 1)
  glm::vec3 localOrigin = glm::vec3(modelMatInv * glm::vec4(ray.rayOrigin, 1.0f));

  // Transform ray direction into local space (use homogeneous coordinate 0)
  glm::vec3 localDirection = glm::normalize(glm::vec3(modelMatInv * glm::vec4(ray.rayDirection, 0.0f)));

  // Compute quadratic equation
  float a, b, c, discriminant;
  vec3 oc = localOrigin;
  a = glm::dot(localDirection, localDirection);
  b = 2.0f * glm::dot(oc, localDirection);
  c = glm::dot(oc, oc) - shape->getRadius() * shape->getRadius();
  discriminant = b * b - 4 * a * c;

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

  // Object space ray hit
  vec3 localHitPoint = localOrigin + t * localDirection;
  vec3 localNormal = glm::normalize(localHitPoint); // For sphere

  // World space ray hit (dividie by w to enter eye space)
  vec4 worldHitPoint4 = modelMat * glm::vec4(localHitPoint, 1.0f);
  vec3 worldHitPoint = glm::vec3(worldHitPoint4) / worldHitPoint4.w;

  // Transform normal to world space
  glm::mat4 invTransModel = glm::transpose(modelMatInv);
  glm::vec4 worldNormal4 = invTransModel * glm::vec4(localNormal, 0.0f);
  glm::vec3 worldNormal = glm::normalize(glm::vec3(worldNormal4));

  // Calc t in world space
  float t_world = glm::length(worldHitPoint - ray.rayOrigin);
  if (glm::dot(ray.rayDirection, worldHitPoint - ray.rayOrigin) < 0.0f) {
    t_world = -t_world;
  }

  // Ray hit
  hit->t = t_world;
  vec3 P = worldHitPoint; // Intersection point
  vec3 N = worldNormal;   // Normal at intersection point
  hit->x = P;
  hit->n = N;

  // Add all lights contribution to the color
  hit->color = calcLightContribution(lights, P, N, shape, ray);
  hit->collision = true;

  return hit;
}

shared_ptr<Hit> computeIntersectionPlane(const Ray &ray, const shared_ptr<Shape> &shape, const mat4 modelMat,
                                         const mat4 modelMatInv, const vector<Light> &lights) {
  shared_ptr<Hit> hit = make_shared<Hit>(); // assume collision is false

  // For an infinite plane, we assume the plane passes through shape->getPosition()
  vec3 planePos = shape->getPosition();
  vec3 N = vec3(0, 1, 0);

  float denom = glm::dot(N, ray.rayDirection);
  // Avoid div by zero
  if (fabs(denom) > 1e-6) {
    // Compute t: distance along ray to intersection
    float t = glm::dot(planePos - ray.rayOrigin, N) / denom;
    if (t >= 0.0f) { // only count intersections in front of the ray
      vec3 P = ray.rayOrigin + t * ray.rayDirection;

      hit->t = t;
      hit->x = P;
      hit->n = N; // constant normal for an infinite plane
      hit->collision = true;

      hit->color = calcLightContribution(lights, P, N, shape, ray);
    }
  }
  return hit;
}

bool isInShadow(shared_ptr<Hit> nearestHit, const vector<Light> &lights, const vector<shared_ptr<Shape>> &shapes) {

  // Shadow test for each light
  float epsilon = 0.001f;
  vec3 shadowOrigin = nearestHit->x + nearestHit->n * epsilon;

  for (const Light &light : lights) {
    Ray shadowRay(shadowOrigin, glm::normalize(light.pos - shadowOrigin));
    float lightDistance = glm::length(light.pos - shadowOrigin);

    for (shared_ptr<Shape> shape : shapes) {
      if (nearestHit->collisionShape == shape) {
        continue; // Don't double count shape that caused initial hit
      }
      // Create Model matrix and apply transformations
      mat4 modelMat = glm::translate(mat4(1.0f), shape->getPosition());
      modelMat = glm::scale(modelMat, shape->getScale());

      // Obatain inv so that ray is in object space
      mat4 modelMatInv = glm::inverse(modelMat);
      shared_ptr<Hit> shadowHit;
      if (shape->getType() == ShapeType::SPHERE || shape->getType() == ShapeType::ELLIPSOID) {
        shadowHit = computeIntersectionSphere(shadowRay, shape, modelMat, modelMatInv, lights);
      }
      if (shape->getType() == ShapeType::PLANE) {
        shadowHit = computeIntersectionPlane(shadowRay, shape, modelMat, modelMatInv, lights);
      }
      if ((shadowHit->collision) && (shadowHit->t > 0) && (shadowHit->t < lightDistance)) {
        return true;
      }
    }
  }
  return false;
}

void genScenePixels(Image &image, int width, int height, shared_ptr<Camera> &camPos, const vector<shared_ptr<Shape>> &shapes,
                    vector<shared_ptr<Hit>> &hits, const vector<Light> &lights, string FILENAME) {
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
        // Create Model matrix and apply transformations
        mat4 modelMat = glm::translate(mat4(1.0f), shape->getPosition());
        modelMat = glm::scale(modelMat, shape->getScale());

        // Obatain inv so that ray is in object space
        mat4 modelMatInv = glm::inverse(modelMat);
        shared_ptr<Hit> curHit;
        if (shape->getType() == ShapeType::SPHERE || shape->getType() == ShapeType::ELLIPSOID) {
          curHit = computeIntersectionSphere(Ray(camPos->getPosition(), rayDir), shape, modelMat, modelMatInv, lights);
        }
        if (shape->getType() == ShapeType::PLANE) {
          curHit = computeIntersectionPlane(Ray(camPos->getPosition(), rayDir), shape, modelMat, modelMatInv, lights);
        }
        if (curHit->collision && curHit->t < nearestToCamT) {
          nearestToCamT = curHit->t;
          finalColor = curHit->color;
          curHit->collisionShape = shape;
          nearestHit = curHit;
        }
      }

      // If hit exists, do shadow test
      if ((nearestHit != nullptr) && isInShadow(nearestHit, lights, shapes)) {
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

void sceneOne(int width, int height, std::vector<std::shared_ptr<Material>> materials,
              std::vector<std::shared_ptr<Shape>> &shapes, std::vector<std::shared_ptr<Hit>> &hits, std::shared_ptr<Camera> &cam,
              std::string FILENAME) {
  Image image(width, height);
  // Shape(ShapeType type, glm::vec3 position, float radius, std::shared_ptr<Material> material)
  shared_ptr<Shape> redSphere =
      make_shared<Shape>(ShapeType::SPHERE, vec3(-0.5f, -1.0f, 1.0f), 1.0f, vec3(1.0f, 1.3f, 1.0f), materials[0]);
  shared_ptr<Shape> greenSphere = make_shared<Shape>(ShapeType::SPHERE, vec3(0.5f, -1.0f, -1.0f), 1.0f, vec3(1.0f), materials[1]);
  shared_ptr<Shape> blueSphere = make_shared<Shape>(ShapeType::SPHERE, vec3(0.0f, 1.0f, 0.0f), 1.0f, vec3(1.0f), materials[2]);
  shapes.push_back(redSphere);
  shapes.push_back(greenSphere);
  shapes.push_back(blueSphere);
  Light worldLight(vec3(-2.0f, 1.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f), 1.0f);
  vector<Light> lights;
  lights.push_back(worldLight);

  genScenePixels(image, width, height, cam, shapes, hits, lights, FILENAME);
}

void sceneThree(int width, int height, std::vector<std::shared_ptr<Material>> materials,
                std::vector<std::shared_ptr<Shape>> &shapes, std::vector<std::shared_ptr<Hit>> &hits,
                std::shared_ptr<Camera> &cam, std::string FILENAME) {
  Image image(width, height);
  shared_ptr<Shape> redEllipsoid =
      make_shared<Shape>(ShapeType::ELLIPSOID, vec3(0.5f, 0.0f, 0.5f), 1.0f, vec3(0.5f, 0.6f, 0.2f), materials[0]);
  shared_ptr<Shape> greenSphere = make_shared<Shape>(ShapeType::SPHERE, vec3(-0.5f, 0.0f, -0.5f), 1.0f, vec3(1.0f), materials[1]);
  // Plane
  shared_ptr<Shape> plane =
      make_shared<Shape>(vec3(0.0f, -1.0f, 0.0f), glm::normalize(vec3(0.0f, 1.0f, 0.0f)), vec3(1.0f), materials[3]);
  shapes.push_back(redEllipsoid);
  shapes.push_back(greenSphere);
  shapes.push_back(plane);
  // Create scene lights
  Light worldLightOne(vec3(1.0f, 2.0f, 2.0f), vec3(1.0f), 0.5f);
  Light worldLightTwo(vec3(-1.0f, 2.0f, -1.0f), vec3(1.0f), 0.5f);
  vector<Light> lights;
  lights.push_back(worldLightOne);
  lights.push_back(worldLightTwo);
  genScenePixels(image, width, height, cam, shapes, hits, lights, FILENAME);
}
