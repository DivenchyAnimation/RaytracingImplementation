#include "HostLauncher.h"

// Define E mat

// E =
//
//     1.5000         0         0    0.3000
//          0    1.4095   -0.5130   -1.5000
//          0    0.5130    1.4095         0
//          0         0         0    1.0000
mat4 E(
	vec4(1.5f, 0.0f, 0.0f, 0.0f), 
	vec4(0.0f, 1.4095f, 0.5130, 0.0f), 
	vec4(0.0f, -0.5130f, 1.4095f, 0.0f), 
	vec4(0.3f, -1.50f, 0.0f, 1.0f));

mat4 IdMat = mat4();

void loadHostShapes(std::vector<GPUShape> &hostShapes, GPUMaterial *materials) {
	// Create device friendly shapes and allocate memory on device
	GPUShape redSphere = GPUShape(vec3(-0.5f, -1.0f, 1.0f), vec3(0.0f), vec3(1.0f), 0.0f, materials[0], false);
	redSphere.data.SPHERE.radius = 1.0f;
	GPUShape greenSphere = GPUShape(vec3(0.5f, -1.0f, -1.0f), vec3(0.0f), vec3(1.0f), 0.0f, materials[1], false);
	greenSphere.data.SPHERE.radius = 1.0f;
	GPUShape blueSphere = GPUShape(vec3(0.0f, 1.0f, 0.0f), vec3(0.0f), vec3(1.0f), 0.0f, materials[2], false);
	blueSphere.data.SPHERE.radius = 1.0f;
	hostShapes.push_back(redSphere);
	hostShapes.push_back(greenSphere);
	hostShapes.push_back(blueSphere);
};


// HA -> Hardware Accelerated
void HAsceneOne(int blocks, int numThreads, unsigned char *&d_pixels, int numPixels, int width, int height, GPUMaterial *materials,
              GPUShape **&device_shapesPtrs, GPULight *&device_lights, GPUCamera *&device_cam, char *FILENAME) {
}
