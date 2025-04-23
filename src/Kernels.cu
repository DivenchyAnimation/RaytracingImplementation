#include "Kernels.h"

using glm::vec3, glm::vec4, glm::mat4;

// Example kernel that fills the png with the color red
__global__ void fillRedKernel(unsigned char *d_pixels, int numPixels) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numPixels) return;
	int base = idx * 3;
	d_pixels[base + 0] = 255; // R
	d_pixels[base + 1] = 0; // G
	d_pixels[base + 2] = 0; // B
}

__device__ bool isInShadow(GPUHit *nearestHit, const Light &light, const GPUShape **shapes, int nShapes) {

  // Shadow test for each light
  float epsilon = 0.001f;
  vec3 shadowOrigin = nearestHit->x + nearestHit->n * epsilon;

  Ray shadowRay(shadowOrigin, glm::normalize(light.pos - shadowOrigin));
  float lightDistance = glm::length(light.pos - shadowOrigin);

  for (int i = 0; i < nShapes; i++) {
    const GPUShape *shape = shapes[i];
    if (shape->type == GPUShapeType::PLANE) {
      continue;
    }
    if (nearestHit->collisionShape == shape) {
      continue; // Don't double count shape that caused initial hit
    }
    // Create Model matrix and apply transformations
    mat4 modelMat = glm::translate(mat4(1.0f), shape->position);
    modelMat = glm::scale(modelMat, shape->scale);

    // Obatain inv so that ray is in object space
    mat4 modelMatInv = glm::inverse(modelMat);
    GPUHit *shadowHit;
    // Create light array to fit signature of function
    Light *lights = (Light*)malloc(1 * sizeof(Light));
    lights[0] = light;
    const Light *lightArray = lights;
    shadowHit = computeIntersection(shape, shadowRay, modelMat, modelMatInv, lightArray);
    // If a collision occurs and the distance is less than the light's, then this light is occluded.
    if (shadowHit && shadowHit->collision && (shadowHit->t > 0) && (shadowHit->t < lightDistance)) {
      return true;
    }
  }
  return false;
}

__device__ glm::vec3 KernerlcalcLightContribution(const Light &light, GPUHit *nearestHit, Ray ray, const GPUShape *shapes, int nShapes) {

}
