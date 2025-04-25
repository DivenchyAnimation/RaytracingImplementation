#include "pch.cuh"
#include "Host.h"

char *FILENAME;
int SCENE;
int width, height;
GPUShape **device_shapesPtrs = NULL;
GPULight *device_lights = NULL;
GPUCamera *cam;
GPUMaterial *materials;

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

	// Create Image
	Image image(width, height);

	// Create materials
	initMaterials(materials);

	// Allocate memory on the device
	unsigned char *d_pixels = NULL;
	size_t numPixels = width * height;
	size_t bufferSize = numPixels * sizeof(unsigned char) * 3;
	cudaMalloc(&d_pixels, bufferSize);

	//// Launch kernel
	//int numThreads = 256;
	//int blocks = (numPixels + numThreads - 1) / numThreads;
	//fillRedKernel <<<blocks, numThreads >>> (d_pixels, numPixels);
	//cudaDeviceSynchronize();

	// Launch kernel for scene , thanks GPT for these two vals
	int numThreads = 256;
	int blocks = (numPixels + numThreads - 1) / numThreads;
	HAsceneOne(blocks, numThreads, d_pixels, numPixels, width, height, materials, device_shapesPtrs, device_lights, cam, FILENAME);
	cudaDeviceSynchronize();

	// 3) Copy back into the Image's vector
	cudaMemcpy(image.getPixels(), d_pixels, bufferSize, cudaMemcpyDeviceToHost);
	cudaFree(d_pixels);

	// 4) Write out
	image.writeToFile("red_from_cuda.png");
	printf("Wrote red_from_cuda.png\n");
	return 0;
}