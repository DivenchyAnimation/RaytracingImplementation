#include "Kernels.h"

// Example kernel that fills the png with the color red
__global__ void fillRedKernel(unsigned char *d_pixels, int numPixels) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numPixels) return;
	int base = idx * 3;
	d_pixels[base + 0] = 255; // R
	d_pixels[base + 1] = 0; // G
	d_pixels[base + 2] = 0; // B
}

HD bool isInShadow(GPUHit *nearestHit, const GPULight &light, const GPUShape **shapes, int nShapes) {

  // Shadow test for each light
  float epsilon = 0.001f;
  vec3 shadowOrigin = nearestHit->x + nearestHit->n * epsilon;

  GPURay shadowRay(shadowOrigin, GPUnormalize(light.pos - shadowOrigin));
  float lightDistance = GPUlength(light.pos - shadowOrigin);

  for (int i = 0; i < nShapes; i++) {
    const GPUShape *shape = shapes[i];
    if (shape->type == GPUShapeType::PLANE) {
      continue;
    }
    if (nearestHit->collisionShape == shape) {
      continue; // Don't double count shape that caused initial hit
    }
    // Create Model matrix and apply transformations
    mat4 modelMat = GPUtranslate(mat4(), shape->position);  
    modelMat = GPUscale(modelMat, shape->scale);

    // Obatain inv so that ray is in object space
    mat4 modelMatInv = GPUinverse(modelMat);
    GPUHit *shadowHit;
    // Create light array to fit signature of function
    GPULight *lights = (GPULight*)malloc(1 * sizeof(GPULight));
    lights[0] = light;
    const GPULight *lightArray = lights;
    shadowHit = computeIntersection(shape, shadowRay, modelMat, modelMatInv, lightArray); 

    // If a collision occurs and the distance is less than the light's, then this light is occluded.
    if (shadowHit && shadowHit->collision && (shadowHit->t > 0) && (shadowHit->t < lightDistance)) {
      return true;
    }
  }
  return false;
}

HD vec3 KernelCalcLightContribution(const GPULight &light, GPUHit *nearestHit, GPURay ray, const GPUShape **shapes, int nShapes) {
  bool isOccluded = isInShadow(nearestHit, light, shapes, nShapes);

  // For now, use binary shadowing.
  float shadowFactor = isOccluded ? 0.0f : 1.0f;
  GPUShape *shape = nearestHit->collisionShape;

  // 2. Compute diffuse shading.
  vec3 L = GPUnormalize(light.pos - nearestHit->x);
  float diff = GPUmax(GPUdot(nearestHit->n, L), 0.0f);
  vec3 diffuse = light.intensity * shape->material->getMaterialKD() * diff;

  // 3. Compute specular shading (Phong model).
  vec3 V = GPUnormalize(ray.rayOrigin - nearestHit->x); // View direction.
  vec3 H = GPUnormalize(L + V);                         // Halfway vector.
  float spec = pow(GPUmax(GPUdot(nearestHit->n, H), 0.0f), shape->material->getMaterialS());
  vec3 specular = light.intensity * shape->material->getMaterialKS() * spec;

  // 4. Return the light's contribution scaled by the shadow factor.
  return shadowFactor * (diffuse + specular);

	return vec3(0.0f, 0.0f, 0.0f);
};


// This should achieve what fillRedKernel does
HD void KernelGenScenePixels(Image &image, int width, int height, GPUCamera *camPos,
     GPUShape **shapes, int nShapes, const GPULight *lights, int nLights, char *FILENAME, int SCENE, mat4 E) {

}
