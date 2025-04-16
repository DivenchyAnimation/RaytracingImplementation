// src/ Routines.cpp
#include "Routines.h"
#include "pch.h"
#include <memory>

using std::vector, std::shared_ptr, std::make_shared, std::string, std::sqrt, glm::vec3, glm::vec4, glm::mat4;

// Helper function
float clamp(float x, float minVal = 0.0f, float maxVal = 1.0f) { return std::max(minVal, std::min(x, maxVal)); }

vec3 calcLightContribution(const Light &light, shared_ptr<Hit> nearestHit, Ray ray, const vector<shared_ptr<Shape>> &shapes) {
  bool isOccluded = isInShadow(nearestHit, light, shapes);

  // For now, use binary shadowing.
  float shadowFactor = isOccluded ? 0.0f : 1.0f;
  shared_ptr<Shape> shape = nearestHit->collisionShape;

  // 2. Compute diffuse shading.
  vec3 L = glm::normalize(light.pos - nearestHit->x);
  float diff = std::max(glm::dot(nearestHit->n, L), 0.0f);
  vec3 diffuse = light.intensity * shape->getMaterial()->getMaterialKD() * diff;

  // 3. Compute specular shading (Phong model).
  vec3 V = glm::normalize(ray.rayOrigin - nearestHit->x); // View direction.
  vec3 H = glm::normalize(L + V);                         // Halfway vector.
  float spec = pow(std::max(glm::dot(nearestHit->n, H), 0.0f), shape->getMaterial()->getMaterialS());
  vec3 specular = light.intensity * shape->getMaterial()->getMaterialKS() * spec;

  // 4. Return the light's contribution scaled by the shadow factor.
  return shadowFactor * (diffuse + specular);
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

// Returns if light is occluded
bool isInShadow(shared_ptr<Hit> nearestHit, const Light &light, const vector<shared_ptr<Shape>> &shapes) {

  // Shadow test for each light
  float epsilon = 0.001f;
  vec3 shadowOrigin = nearestHit->x + nearestHit->n * epsilon;

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
    shadowHit = shape->computeIntersection(shadowRay, modelMat, modelMatInv, vector<Light>{light});
    // If a collision occurs and the distance is less than the light's, then this light is occluded.
    if (shadowHit && shadowHit->collision && (shadowHit->t > 0) && (shadowHit->t < lightDistance)) {
      return true;
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
        curHit = shape->computeIntersection(Ray(camPos->getPosition(), rayDir), modelMat, modelMatInv, lights);
        if (curHit->collision && curHit->t < nearestToCamT) {
          nearestToCamT = curHit->t;
          curHit->collisionShape = shape;
          nearestHit = curHit;
        }
      }

      // If hit exists, do shadow test
      if ((nearestHit != nullptr)) {
        // init color to ambient, then add light contributions
        vec3 totalLight = nearestHit->collisionShape->getMaterial()->getMaterialKA();
        for (const Light &light : lights) {
          totalLight += calcLightContribution(light, nearestHit, Ray(camPos->getPosition(), rayDir), shapes);
        }
        finalColor = totalLight;
      }

      // Convert color components from [0,1] to [0,255].

      finalColor.r = clamp(finalColor.r);
      finalColor.g = clamp(finalColor.g);
      finalColor.b = clamp(finalColor.b);

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
  // Sphere(glm::vec3 position, float radius, float scale, float rotAngle, std::shared_ptr<Material> material) : Shape() {
  shared_ptr<Shape> redSphere = make_shared<Sphere>(vec3(-0.5f, -1.0f, 1.0f), 1.0f, 1.0f, 0.0f, materials[0]);
  shared_ptr<Shape> greenSphere = make_shared<Sphere>(vec3(0.5f, -1.0f, -1.0f), 1.0f, 1.0f, 0.0f, materials[1]);
  shared_ptr<Shape> blueSphere = make_shared<Sphere>(vec3(0.0f, 1.0f, 0.0f), 1.0f, 1.0f, 0.0f, materials[2]);
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
  // Ellipsoid(glm::vec3 position, float radius, glm::vec3 scale, float rotAngle, std::shared_ptr<Material> material) : Shape() {
  shared_ptr<Shape> redEllipsoid =
      make_shared<Ellipsoid>(vec3(0.5f, 0.0f, 0.5f), 1.0f, vec3(0.5f, 0.6f, 0.2f), 0.0f, materials[0]);
  shared_ptr<Shape> greenSphere = make_shared<Sphere>(vec3(-0.5f, 0.0f, -0.5f), 1.0f, 1.0f, 0.0f, materials[1]);
  // Plane
  //Plane(glm::vec3 position, glm::vec3 normal, float scale, float rotAngle, std::shared_ptr<Material> material) : Shape() {
  shared_ptr<Shape> plane =
      make_shared<Plane>(vec3(0.0f, -1.0f, 0.0f), glm::normalize(vec3(0.0f, 1.0f, 0.0f)), 1.0f, 0.0f, materials[3]);
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
