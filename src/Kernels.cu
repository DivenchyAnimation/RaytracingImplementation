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



// Beginning of helpers
HD mat4 buildMVMat(GPUShape *s, mat4 E) {
    mat4 modelMat = GPUtranslate(mat4(), s->position);
    modelMat = modelMat * E;

    return modelMat;
}

HD bool isInShadow(GPUHit &nearestHit, const GPULight &light, GPUShape **shapes, int nShapes) {

  // Shadow test for each light
  float epsilon = 0.001f;
  vec3 shadowOrigin = nearestHit.x + nearestHit.n * epsilon;

  GPURay shadowRay(shadowOrigin, GPUnormalize(light.pos - shadowOrigin));
  float lightDistance = GPUlength(light.pos - shadowOrigin);

  for (int i = 0; i < nShapes; i++) {
    const GPUShape *shape = shapes[i];
    if (shape->type == GPUShapeType::PLANE) {
      continue;
    }
    if (nearestHit.collisionShape == shape) {
      continue; // Don't double count shape that caused initial hit
    }
    // Create Model matrix and apply transformations
    mat4 modelMat = GPUtranslate(mat4(), shape->position);  
    modelMat = GPUscale(modelMat, shape->scale);

    // Obatain inv so that ray is in object space
    mat4 modelMatInv = GPUinverse(modelMat);
    GPUHit shadowHit = computeIntersection(shape, shadowRay, modelMat, modelMatInv); 

    // If a collision occurs and the distance is less than the light's, then this light is occluded.
    if (shadowHit.collision && (shadowHit.t > 0) && (shadowHit.t < lightDistance)) {
      return true;
    }
  }
  return false;
}

HD vec3 KernelCalcLightContribution(const GPULight &light, GPUHit &nearestHit, GPURay ray, GPUShape **shapes, int nShapes) {
  bool isOccluded = isInShadow(nearestHit, light, shapes, nShapes);

  // For now, use binary shadowing.
  float shadowFactor = isOccluded ? 0.0f : 1.0f;
  GPUShape *shape = nearestHit.collisionShape;

  // 2. Compute diffuse shading.
  vec3 L = GPUnormalize(light.pos - nearestHit.x);
  float diff = GPUMax(GPUdot(nearestHit.n, L), 0.0f);
  vec3 diffuse = light.intensity * shape->material.getMaterialKD() * diff;

  // 3. Compute specular shading (Phong model).
  vec3 V = GPUnormalize(ray.rayOrigin - nearestHit.x); // View direction.
  vec3 H = GPUnormalize(L + V);                         // Halfway vector.
  float spec = pow(GPUMax(GPUdot(nearestHit.n, H), 0.0f), shape->material.getMaterialS());
  vec3 specular = light.intensity * shape->material.getMaterialKS() * spec;

  // 4. Return the light's contribution scaled by the shadow factor.
  return shadowFactor * (diffuse + specular);

	return vec3(0.0f, 0.0f, 0.0f);
};

HD GPURay GPUGenRayForPixel(int x, int y, int width, int height, GPUCamera *cam) {
  float fov = cam->getFOVY();
  float aspect = float(width) / float(height);
  float tanY = tan(fov * 0.5f);
  float tanX = tanY * aspect;

  // Help from ChatGPT
  // Map pixel (x,y) to norm coords
  float u = (((x + 0.5f) / width) * 2.0f - 1.0f) * tanX;
  float v = (((y + 0.5f) / height) * 2.0f - 1.0f) * tanY;

  vec3 camPos = cam->getPosition();
  vec3 forward = GPUnormalize(cam->getTarget() - camPos);
  vec3 right = GPUnormalize(GPUcross(forward, cam->getWorldUp()));
  vec3 up = GPUcross(right, forward);

  vec3 rayDirWorld = GPUnormalize(u * right + v * up + 1.0f * forward);

  return GPURay(camPos, rayDirWorld);
}

HD vec3 GPUTraceRay(GPURay ray, GPUHit &nearestHit, GPUShape **shapes, int nShapes, GPULight *lights, int nLights, int depth, mat4 E) {
    // No more bounces
    if (depth <= 0) {
        return vec3(0.0f);
    }

    // Init dummy t
    float nearestToCamT = GPU_MAX_FLT();
    vec3 finalColor(0.0f);  // Init to black background

    // Check intersections on each shape
    for (int i = 0; i < nShapes; i++) {
        GPUShape *shape = shapes[i];
        mat4 modelMat = buildMVMat(shape, E);
        // Inv for obj space
        mat4 modelMatInv = GPUinverse(modelMat);

        GPUHit curHit;
        curHit = computeIntersection(shape, ray, modelMat, modelMatInv);
        if (curHit.collision && curHit.t < nearestToCamT) {
            nearestToCamT = curHit.t;
            curHit.collisionShape = shape;
            nearestHit = curHit;
        }
    }

    // If hit exists, do shadows
    if (nearestHit.collision == true) {
        // init color to ambient
        finalColor = nearestHit.collisionShape->material.getMaterialKA();
        for (int i = 0; i < nLights; i++) {
            GPULight curLight = lights[i];
            finalColor += KernelCalcLightContribution(curLight, nearestHit, ray, shapes, nShapes);
        }
    }
    else {
        return finalColor;
    }

    // Reflection step
    float kr = nearestHit.collisionShape->material.getMaterialReflectivity();
    if (kr > 0.0f) {
        // reflect direction about the normal
        vec3 I = GPUnormalize(ray.rayDirection);
        vec3 N = nearestHit.n;
        vec3 R = I - 2.0f * GPUdot(I, N) * N;
        // offset origin to avoid self-reflection
        vec3 origin = nearestHit.x + N * 0.001f;
        GPURay reflectRay(origin, R);
        vec3 reflCol = GPUTraceRay(reflectRay, nearestHit, shapes, nShapes, lights, nLights, depth, E);
        // blend: local*(1-kr) + reflection*kr
        finalColor = GPUMix(finalColor, reflCol, kr);
    }

    return finalColor;
};

// This should achieve what fillRedKernel does (aka fill d_pixels)
__global__ void KernelGenScenePixels(unsigned char *d_pixels, int numPixels, int width, int height, GPUCamera *cam,
     GPUShape **shapes, int nShapes, GPULight *lights, int nLights, int SCENE, mat4 E) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numPixels) return;

    int x = idx % width;
    int y = idx / height;

	GPURay ray = GPUGenRayForPixel(x, y, width, height, cam);
    GPUHit nearestHit;

	int depth;
	if (SCENE == 4) {
		depth = 2;
	} else {
		depth = 5;
	}
	vec3 finalColor = GPUTraceRay(ray, nearestHit, shapes, nShapes, lights, nLights, depth, E);
	// Clamp vals to 0 to 1 to get percent val 0 - 255
    d_pixels[3 * idx + 0] = (unsigned char)(GPUClampf(finalColor.x)* 255);
    d_pixels[3 * idx + 1] = (unsigned char)(GPUClampf(finalColor.y)* 255);
    d_pixels[3 * idx + 2] = (unsigned char)(GPUClampf(finalColor.z)* 255);

    // Sanity check
	//int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//if (idx >= numPixels) return;
	//int base = idx * 3;
	//d_pixels[base + 0] = 255; // R
	//d_pixels[base + 1] = 255; // G
	//d_pixels[base + 2] = 255; // B

}
