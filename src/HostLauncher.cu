#include "Host.h"

// Define E mat

// E =
//
//     1.5000         0         0    0.3000
//          0    1.4095   -0.5130   -1.5000
//          0    0.5130    1.4095         0
//          0         0         0    1.0000
mat4 E(
	vec4(1.5f, 0.0f, 0.0f, 0.3f), 
	vec4(0.0f, 1.4095f, -0.5130, -1.500), 
	vec4(0.0f, 0.5130f, 1.4095f, 0.0f), 
	vec4(0.0f, 0.0f, 0.0, 1.0f));

mat4 IdMat = mat4();

// HA -> Hardware Accelerated
void HAsceneOne(int blocks, int numThreads, unsigned char *d_pixels, int numPixels, int width, int height, GPUMaterial *materials,
              GPUShape **device_shapesPtrs, GPULight *device_lights, GPUCamera *cam, char *FILENAME) {

	// Sphere(glm::vec3 position, float radius, float scale, float rotAngle, std::shared_ptr<Material> material) : Shape() {
	//shared_ptr<Shape> redSphere = make_shared<Sphere>(glm::vec3(-0.5f, -1.0f, 1.0f), 1.0f, 1.0f, 0.0f, materials[0]);
	//shared_ptr<Shape> greenSphere = make_shared<Sphere>(glm::vec3(0.5f, -1.0f, -1.0f), 1.0f, 1.0f, 0.0f, materials[1]);
	//shared_ptr<Shape> blueSphere = make_shared<Sphere>(glm::vec3(0.0f, 1.0f, 0.0f), 1.0f, 1.0f, 0.0f, materials[2]);
	//shapes.push_back(redSphere);
	//shapes.push_back(greenSphere);
	//shapes.push_back(blueSphere);
	//Light worldLight(glm::vec3(-2.0f, 1.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f), 1.0f);
	//vector<Light> lights;
	//lights.push_back(worldLight);


	// Create device friendly shapes
	device_shapesPtrs = (GPUShape **)malloc(3 * sizeof(GPUShape *));
	GPUShape *redSphere = new GPUShape(vec3(-0.5f, -1.0f, 1.0f), vec3(0.0f), vec3(1.0f), 0.0f, materials[0], false);
	GPUShape *greenSphere = new GPUShape(vec3(-0.5f, -1.0f, 1.0f), vec3(0.0f), vec3(1.0f), 0.0f, materials[1], false);
	GPUShape *blueSphere = new GPUShape(vec3(-0.5f, -1.0f, 1.0f), vec3(0.0f), vec3(1.0f), 0.0f, materials[2], false);
	device_shapesPtrs[0] = redSphere;
	device_shapesPtrs[1] = greenSphere;
	device_shapesPtrs[2] = blueSphere;

	// Create device friendly world light, one in this case
	device_lights = (GPULight *)malloc(sizeof(GPULight));
	GPULight worldLight = GPULight(vec3(-2.0f, 1.0f, 1.0f), vec3(1.0f), 1.0f);
	device_lights[0] = worldLight;

	// Create device friendly camera, and to device friendly structs
	GPUCamera *gpuCam = (GPUCamera *)malloc(sizeof(GPUCamera));
	*gpuCam = GPUCamera();
	
	KernelGenScenePixels <<<blocks, numThreads >>> (d_pixels, numPixels, width, height, gpuCam, device_shapesPtrs, 3, device_lights, 1, 1, IdMat);
}
