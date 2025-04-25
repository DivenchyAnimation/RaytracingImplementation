#include "pch.cuh"
#include "Host.h"
#include "HostLauncher.h"

char *FILENAME;
int SCENE;
int width, height;
std::vector<GPUShape> hostShapes;
std::vector<GPULight> hostLights;
GPUShape *device_shapes;
GPUShape **device_shapesPtrs;
GPULight *device_lights;
GPUCamera *device_cam;
GPUMaterial *materials;
GPUMaterial *device_materials;

extern __global__ void fillRedKernel(unsigned char *, int);

int main(int arc, char** argv) {

	if (arc != 4) {
		printf("Usage: ./A6 <SCENE> <IMAGE SIZE> <OUTFILENAME>");
		return 1;
	}

	// Init user input
	SCENE = atoi(argv[1]);
	width = atoi(argv[2]);
	height = width;
	FILENAME = argv[3];
	mat4 IdMat = mat4();

	// Create Image
	Image image(width, height);

	// Create materials and allocate memory on device
	GPUMaterial hostMaterials[6];
	initMaterials(hostMaterials);
	device_materials = nullptr;
	cudaMalloc(&device_materials, 6 * sizeof(GPUMaterial));
	cudaMemcpy(device_materials, hostMaterials, 6 * sizeof(GPUMaterial), cudaMemcpyHostToDevice);

	// Create shape device memory
	loadHostShapes(hostShapes, hostMaterials);
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

	// Allocate memory on the device
	unsigned char *d_pixels = NULL;
	size_t numPixels = width * height;
	size_t bufferSize = numPixels * sizeof(unsigned char) * 3;
	cudaMalloc(&d_pixels, bufferSize);
	// Launch kernel for scene , thanks GPT for these two vals
	int numThreads = 256;
	int blocks = (numPixels + numThreads - 1) / numThreads;
	//HAsceneOne(blocks, numThreads, d_pixels, numPixels, width, height, hostMaterials, device_shapesPtrs, device_lights, device_cam, FILENAME);
	KernelGenScenePixels <<<blocks, numThreads >>> (d_pixels, numPixels, width, height, device_cam, device_shapesPtrs, 3, device_lights, 1, 1, IdMat);
	cudaDeviceSynchronize();

	// 3) Copy back into the Image's vector
	cudaMemcpy(image.getPixels(), d_pixels, bufferSize, cudaMemcpyDeviceToHost);
	cudaFree(d_pixels);

	// 4) Write out
	image.writeToFile(FILENAME);
	printf("Wrote %s\n", FILENAME);
	return 0;
}