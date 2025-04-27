#include "pch.cuh"
#include "Host.h"
#include "HostLauncher.h"

char *FILENAME;
int SCENE;
int width, height;
int nShapes;
int nLights;
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

	// Allocate memory on the device
	unsigned char *d_pixels = NULL;
	size_t numPixels = width * height;
	size_t bufferSize = numPixels * sizeof(unsigned char) * 3;
	cudaMalloc(&d_pixels, bufferSize);
	// Launch kernel for scene , thanks GPT for these two vals
	int numThreads = 256;
	int blocks = (numPixels + numThreads - 1) / numThreads;

	switch (SCENE) {
	case 1:
	case 2: {
		HAsceneOne(blocks, numThreads, d_pixels, numPixels, width, height, device_materials, materials, device_shapes, device_shapesPtrs, nShapes, device_lights, nLights, device_cam, IdMat);
		break;
	};
	case 3: {
		HAsceneThree(blocks, numThreads, d_pixels, numPixels, width, height, device_materials, materials, device_shapes, device_shapesPtrs, nShapes, device_lights, nLights, device_cam, IdMat);
		break;
	};
	case 4:
	case 5: {
		HAsceneReflections(blocks, numThreads, d_pixels, numPixels, width, height, device_materials, materials, device_shapes, device_shapesPtrs, nShapes, device_lights, nLights, device_cam, IdMat);
	};
	}
	KernelGenScenePixels <<<blocks, numThreads >>> (d_pixels, numPixels, width, height, device_cam, device_shapesPtrs, nShapes, device_lights, nLights, SCENE, IdMat);
	cudaDeviceSynchronize();

	// 3) Copy back into the Image's vector
	cudaMemcpy(image.getPixels(), d_pixels, bufferSize, cudaMemcpyDeviceToHost);
	cudaFree(d_pixels);

	// 4) Write out
	image.writeToFile(FILENAME);
	printf("Wrote %s\n", FILENAME);
	return 0;
}