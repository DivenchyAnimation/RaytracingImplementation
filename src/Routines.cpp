// src/ Routines.cpp
#include "Routines.h"

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

// Compute intersections of ray with shape in order to create the shading
vec3 computeIntersection(const Ray &ray, const shared_ptr<Shape> &shape, const Light &light) {
  // Compute quadratic equation
  vec3 oc = ray.rayOrigin - shape->getPosition();
  float a = glm::dot(ray.rayDirection, ray.rayDirection);
  float b = 2.0f * glm::dot(oc, ray.rayDirection);
  float c = glm::dot(oc, oc) - shape->getRadius() * shape->getRadius();
  float discriminant = b * b - 4 * a * c;

  // If discriminant is negative, no collision with shape
  if (discriminant < 0) {
    return vec3(0.0f);
  }

  // Else, colllision so solve for t (intersections)
  float sqrtDiscriminant = sqrt(discriminant);
  float t = (-b - sqrtDiscriminant) / (2.0f * a);

  // Behind camera, count as miss
  if (t < 0) {
    t = (-b + sqrtDiscriminant) / (2.0f * a);
    return vec3(0.0f);
  }

  vec3 P = ray.rayOrigin + t * ray.rayDirection;     // Intersection point
  vec3 N = glm::normalize(P - shape->getPosition()); // Normal at intersection point
  vec3 L = glm::normalize(light.pos - P);            // Light direction

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
  return color;
}

void genPixelsSceneOne(Image &image, int width, int height, shared_ptr<Camera> &camPos, const shared_ptr<Shape> &shape,
                       const Light &light, string FILENAME) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      vec3 rayDir = genRayForPixel(x, y, width, height, camPos->getFOVY());

      // Get shading
      vec3 color = computeIntersection(Ray(camPos->getPosition(), rayDir), shape, light);
      // Convert color components from [0,1] to [0,255].
      unsigned char rVal = static_cast<unsigned char>(color.r * 255);
      unsigned char gVal = static_cast<unsigned char>(color.g * 255);
      unsigned char bVal = static_cast<unsigned char>(color.b * 255);

      // Set the pixel color in the image.
      image.setPixel(x, y, rVal, gVal, bVal);
    }
  }

  // Write to image
  image.writeToFile(FILENAME);
  std::cout << "Image written to " << FILENAME << std::endl;
}

void taskOne(int width, int height, shared_ptr<Camera> &cam, string FILENAME) {
  Image image(width, height);
  shared_ptr<Shape> shape = make_shared<Shape>();
  Light worldLight(vec3(-2.0f, 1.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f), 1.0f);

  genPixelsSceneOne(image, width, height, cam, shape, worldLight, FILENAME);
}
