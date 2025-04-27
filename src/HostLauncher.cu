#include "HostLauncher.h"

// Define E mat

// E =
//
//     1.5000         0         0    0.3000
//          0    1.4095   -0.5130   -1.5000
//          0    0.5130    1.4095         0
//          0         0         0    1.0000
//mat4 E(
//	vec4(1.5f, 0.0f, 0.0f, 0.0f), 
//	vec4(0.0f, 1.4095f, 0.5130, 0.0f), 
//	vec4(0.0f, -0.5130f, 1.4095f, 0.0f), 
//	vec4(0.3f, -1.50f, 0.0f, 1.0f));
//
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

void loadHostShapesThree(std::vector<GPUShape> &hostShapes, GPUMaterial *materials) {
	GPUShape redEllipsoid = GPUShape(vec3(0.5f, 0.0f, 0.5f), vec3(0.0f), vec3(0.5f, 0.6f, 0.2f), 0.0f, materials[0], false);
	redEllipsoid.data.ELLIPSOID.radius = 1.0f;
	redEllipsoid.type = GPUShapeType::ELLIPSOID;
	GPUShape greenSphere = GPUShape(vec3(-0.5f, 0.0f, -0.5f), vec3(0.0f), vec3(1.0f), 0.0f, materials[1], false);
	greenSphere.data.SPHERE.radius = 1.0f;
	GPUShape plane = GPUShape(vec3(0.0f, -1.0f, 0.0f), vec3(0.0f), vec3(1.0f), 0.0f, materials[3], false);
	plane.data.PLANE.normal = vec3(GPUnormalize(vec3(0.0f, 1.0f, 0.0f)));
	plane.type = GPUShapeType::PLANE;
	hostShapes.push_back(redEllipsoid);
	hostShapes.push_back(greenSphere);
	hostShapes.push_back(plane);
};

void loadHostShapesReflections(std::vector<GPUShape> &hostShapes, GPUMaterial *materials) {
  // Make shapes and push them to vector
  GPUShape redSphere = GPUShape(vec3(0.5f, -0.7f, 0.5f), vec3(0.0f), vec3(0.3f), 0.0f, materials[0], false);
  redSphere.data.SPHERE.radius = 1.0f;
  GPUShape blueSphere = GPUShape(vec3(1.0f, -0.7f, 0.0f), vec3(0.0f), vec3(0.3f), 0.0f, materials[2], false);
  blueSphere.data.SPHERE.radius = 1.0f;
  GPUShape floor = GPUShape(vec3(0.0f, -1.0f, 0.0f), vec3(0.0f), vec3(1.0f), 0.0f, materials[3], false);
  floor.type = GPUShapeType::PLANE;
  floor.data.PLANE.normal = vec3(0.0f, -1.0f, 0.0f);
  floor.rotation = vec3(1.0f, 0.0f, 0.0f);
  floor.rotationAngle = 90.0f;
  GPUShape wall = GPUShape(vec3(0.0f, 0.0f, -3.0f), vec3(0.0f), vec3(1.0f), 0.0f, materials[3], false);
  wall.type = GPUShapeType::PLANE;
  // Rotate wall
  wall.data.PLANE.normal = vec3(0.0f, 1.0f, 0.0f);
  wall.rotation = vec3(1.0f, 0.0f, 0.0f);
  wall.rotationAngle = 90.0f;
  GPUShape reflSphere = GPUShape(vec3(-0.5f, 0.0f, -0.5f), vec3(0.0f), vec3(1.0f), 0.0f, materials[4], true);
  reflSphere.data.SPHERE.radius = 1.0f;
  GPUShape reflSphereTwo = GPUShape(vec3(1.5f, 0.0f, -1.5f), vec3(0.0f), vec3(1.0f), 0.0f, materials[4], true);
  reflSphereTwo.data.SPHERE.radius = 1.0f;
  hostShapes.push_back(redSphere);
  hostShapes.push_back(blueSphere);
  hostShapes.push_back(floor);
  hostShapes.push_back(wall);
  hostShapes.push_back(reflSphere);
  hostShapes.push_back(reflSphereTwo);
};


// Lights loaders 
void loadHostLightsThree(std::vector<GPULight> &hostLights) {
	// Create scene lights
	GPULight worldLightOne(vec3(1.0f, 2.0f, 2.0f), vec3(1.0f), 0.5f);
	GPULight worldLightTwo(vec3(-1.0f, 2.0f, -1.0f), vec3(1.0f), 0.5f);
	hostLights.push_back(worldLightOne);
	hostLights.push_back(worldLightTwo);
};

void loadHostLightsReflections(std::vector<GPULight> &hostLights) {
  // Make lights and push them to vector
  GPULight worldLight = GPULight(vec3(-1.0f, 2.0f, 1.0f), vec3(1.0f), 0.5f);
  GPULight worldLightTwo = GPULight(vec3(0.5f, -0.5f, 0.0f), vec3(1.0f), 0.5f);
  hostLights.push_back(worldLight);
  hostLights.push_back(worldLightTwo);
}

// HA -> Hardware Accelerated
void HAsceneOne(int blocks, int numThreads, unsigned char *&d_pixels, int numPixels, int width, int height, GPUMaterial *&device_materials, GPUMaterial *&materials,
    GPUShape *&device_shapes, GPUShape **&device_shapesPtrs, int &nShapes, GPULight *&device_lights, int &nLights, GPUCamera *&device_cam, mat4 &E) {
	std::vector<GPUShape> hostShapes;
	std::vector<GPULight> hostLights;

	// Create materials and allocate memory on device
	GPUMaterial hostMaterials[6];
	initMaterials(hostMaterials);
	device_materials = nullptr;
	cudaMalloc(&device_materials, 6 * sizeof(GPUMaterial));
	cudaMemcpy(device_materials, hostMaterials, 6 * sizeof(GPUMaterial), cudaMemcpyHostToDevice);

	// Create shape device memory
	loadHostShapes(hostShapes, hostMaterials);
	nShapes = hostShapes.size();
	device_shapes = nullptr;
	cudaMalloc(&device_shapes, hostShapes.size() * sizeof(GPUShape));
	cudaMemcpy(device_shapes, hostShapes.data(), hostShapes.size() * sizeof(GPUShape), cudaMemcpyHostToDevice);

	// Array of pointers
	std::vector<GPUShape *> hostShapesPtrs(hostShapes.size());
	for (size_t i = 0; i < hostShapes.size(); i++) {
		hostShapesPtrs[i] = device_shapes + i;
	}
	// Copy to device
	device_shapesPtrs = nullptr;
	cudaMalloc(&device_shapesPtrs, hostShapesPtrs.size() * sizeof(GPUShape *));
	cudaMemcpy(device_shapesPtrs, hostShapesPtrs.data(), hostShapesPtrs.size() * sizeof(GPUShape *), cudaMemcpyHostToDevice);

	// Create device friendly world light, one in this case and allocate memory on device
	GPULight worldLight = GPULight(vec3(-2.0f, 1.0f, 1.0f), vec3(1.0f), 1.0f);
	hostLights.push_back(worldLight);
	nLights = hostLights.size();
	device_lights = nullptr;
	cudaMalloc(&device_lights, hostLights.size() * sizeof(GPULight));
	cudaMemcpy(device_lights, hostLights.data(), hostLights.size() * sizeof(GPULight), cudaMemcpyHostToDevice);

	// Create device friendly camera, and to device friendly structs
	GPUCamera hostCamera;
	hostCamera.position = (vec3(0.0f, 0.0f, 5.0f));
	hostCamera.setTarget(vec3(0.0f)); // Look at origin
	hostCamera.setFOV(GPUradians(60.0f));
	hostCamera.worldUp = vec3(0.0f, 1.0f, 0.0f);
	device_cam = nullptr;
	cudaMalloc(&device_cam, sizeof(GPUCamera));
	cudaMemcpy(device_cam, &hostCamera, sizeof(GPUCamera), cudaMemcpyHostToDevice);
}

void HAsceneThree(int blocks, int numThreads, unsigned char *&d_pixels, int numPixels, int width, int height, GPUMaterial *&device_materials, GPUMaterial *&materials,
    GPUShape *&device_shapes, GPUShape **&device_shapesPtrs, int &nShapes, GPULight *&device_lights, int &nLights, GPUCamera *&device_cam, mat4 &E) {
	std::vector<GPUShape> hostShapes;
	std::vector<GPULight> hostLights;

	// Create materials and allocate memory on device
	GPUMaterial hostMaterials[6];
	initMaterials(hostMaterials);
	device_materials = nullptr;
	cudaMalloc(&device_materials, 6 * sizeof(GPUMaterial));
	cudaMemcpy(device_materials, hostMaterials, 6 * sizeof(GPUMaterial), cudaMemcpyHostToDevice);

	// Create shape device memory
	loadHostShapesThree(hostShapes, hostMaterials);
	nShapes = hostShapes.size();
	device_shapes = nullptr;
	cudaMalloc(&device_shapes, hostShapes.size() * sizeof(GPUShape));
	cudaMemcpy(device_shapes, hostShapes.data(), hostShapes.size() * sizeof(GPUShape), cudaMemcpyHostToDevice);

	// Array of pointers
	std::vector<GPUShape *> hostShapesPtrs(hostShapes.size());
	for (size_t i = 0; i < hostShapes.size(); i++) {
		hostShapesPtrs[i] = device_shapes + i;
	}
	// Copy to device
	device_shapesPtrs = nullptr;
	cudaMalloc(&device_shapesPtrs, hostShapesPtrs.size() * sizeof(GPUShape *));
	cudaMemcpy(device_shapesPtrs, hostShapesPtrs.data(), hostShapesPtrs.size() * sizeof(GPUShape *), cudaMemcpyHostToDevice);

	// Create device friendly world light, one in this case and allocate memory on device
	loadHostLightsThree(hostLights);
	nLights = hostLights.size();
	device_lights = nullptr;
	cudaMalloc(&device_lights, hostLights.size() * sizeof(GPULight));
	cudaMemcpy(device_lights, hostLights.data(), hostLights.size() * sizeof(GPULight), cudaMemcpyHostToDevice);

	// Create device friendly camera, and to device friendly structs
	GPUCamera hostCamera;
	hostCamera.position = (vec3(0.0f, 0.0f, 5.0f));
	hostCamera.setTarget(vec3(0.0f)); // Look at origin
	hostCamera.setFOV(GPUradians(60.0f));
	hostCamera.worldUp = vec3(0.0f, 1.0f, 0.0f);
	device_cam = nullptr;
	cudaMalloc(&device_cam, sizeof(GPUCamera));
	cudaMemcpy(device_cam, &hostCamera, sizeof(GPUCamera), cudaMemcpyHostToDevice);
}

void HAsceneReflections(int blocks, int numThreads, unsigned char *&d_pixels, int numPixels, int width, int height, GPUMaterial *&device_materials, GPUMaterial *&materials,
	GPUShape *&device_shapes, GPUShape **&device_shapesPtrs, int &nShapes, GPULight *&device_lights, int &nLights, GPUCamera *&device_cam, mat4 &E) {
	std::vector<GPUShape> hostShapes;
	std::vector<GPULight> hostLights;

	// Create materials and allocate memory on device
	GPUMaterial hostMaterials[6];
	initMaterials(hostMaterials);
	device_materials = nullptr;
	cudaMalloc(&device_materials, 6 * sizeof(GPUMaterial));
	cudaMemcpy(device_materials, hostMaterials, 6 * sizeof(GPUMaterial), cudaMemcpyHostToDevice);

	// Create shape device memory
	loadHostShapesReflections(hostShapes, hostMaterials);
	nShapes = hostShapes.size();
	device_shapes = nullptr;
	cudaMalloc(&device_shapes, hostShapes.size() * sizeof(GPUShape));
	cudaMemcpy(device_shapes, hostShapes.data(), hostShapes.size() * sizeof(GPUShape), cudaMemcpyHostToDevice);

	// Array of pointers
	std::vector<GPUShape *> hostShapesPtrs(hostShapes.size());
	for (size_t i = 0; i < hostShapes.size(); i++) {
		hostShapesPtrs[i] = device_shapes + i;
	}
	// Copy to device
	device_shapesPtrs = nullptr;
	cudaMalloc(&device_shapesPtrs, hostShapesPtrs.size() * sizeof(GPUShape *));
	cudaMemcpy(device_shapesPtrs, hostShapesPtrs.data(), hostShapesPtrs.size() * sizeof(GPUShape *), cudaMemcpyHostToDevice);

	// Create device friendly world light, one in this case and allocate memory on device
	loadHostLightsReflections(hostLights);
	nLights = hostLights.size();
	device_lights = nullptr;
	cudaMalloc(&device_lights, hostLights.size() * sizeof(GPULight));
	cudaMemcpy(device_lights, hostLights.data(), hostLights.size() * sizeof(GPULight), cudaMemcpyHostToDevice);

	// Create device friendly camera, and to device friendly structs
	GPUCamera hostCamera;
	hostCamera.position = (vec3(0.0f, 0.0f, 5.0f));
	hostCamera.setTarget(vec3(0.0f)); // Look at origin
	hostCamera.setFOV(GPUradians(60.0f));
	hostCamera.worldUp = vec3(0.0f, 1.0f, 0.0f);
	device_cam = nullptr;
	cudaMalloc(&device_cam, sizeof(GPUCamera));
	cudaMemcpy(device_cam, &hostCamera, sizeof(GPUCamera), cudaMemcpyHostToDevice);

}
