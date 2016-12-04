#include "Image.h"
#include "PPM.h"

#include <cstdio>
#include <cassert>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

static void CheckCudaErrorAux(const char *, unsigned, const char *,
		cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char *file, unsigned line,
		const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
			<< err << ") at " << file << ":" << line << std::endl;
	exit(1);
}

// useful defines
#define Mask_width 5
#define Mask_radius Mask_width / 2
#define TILE_WIDTH 16
#define w (TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))

__global__ void convolution(float *I, const float *__restrict__ M, float *P,
		int channels, int width, int height) {
	__shared__ float N_ds[w][w];
	int k;
	for (k = 0; k < channels; k++) {
		// First batch loading
		int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
		int destY = dest / w;
		int destX = dest % w;
		int srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius;
		int srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius;
		int src = (srcY * width + srcX) * channels + k;
		if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
			N_ds[destY][destX] = I[src];
		} else {
			N_ds[destY][destX] = 0;
		}

		// Second batch loading
		dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
		destY = dest / w;
		destX = dest % w;
		srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius;
		srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius;
		src = (srcY * width + srcX) * channels + k;
		if (destY < w) {
			if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
				N_ds[destY][destX] = I[src];
			} else {
				N_ds[destY][destX] = 0;
			}
		}
		__syncthreads();

		float accum = 0;
		int y, x;
		for (y = 0; y < Mask_width; y++) {
			for (x = 0; x < Mask_width; x++) {
				accum += N_ds[threadIdx.y + y][threadIdx.x + x]
						* M[y * Mask_width + x];
			}
		}
		y = blockIdx.y * TILE_WIDTH + threadIdx.y;
		x = blockIdx.x * TILE_WIDTH + threadIdx.x;
		if (y < height && x < width)
			P[(y * width + x) * channels + k] = clamp(accum);
		__syncthreads();
	}
}


// simple test to read/write PPM images, and process Image_t data
void test_images() {
	Image_t* inputImg = PPM_import("computer_programming.ppm");
	for (int i = 0; i < 300; i++) {
		Image_setPixel(inputImg, i, 100, 0, float(i) / 300);
		Image_setPixel(inputImg, i, 100, 1, float(i) / 300);
		Image_setPixel(inputImg, i, 100, 2, float(i) / 200);
	}
	PPM_export("test_output.ppm", inputImg);
	Image_t* newImg = PPM_import("test_output.ppm");
	inputImg = PPM_import("computer_programming.ppm");
	if (Image_is_same(inputImg, newImg))
		printf("Same images\n");
	else
		printf("Different images\n");
}

int main() {
	const int maskRows = 5;
	const int maskColumns = 5;
	int imageChannels;
	int imageWidth;
	int imageHeight;
	Image_t* inputImage;
	Image_t* outputImage;
	float *hostInputImageData;
	float *hostOutputImageData;
	float *deviceInputImageData;
	float *deviceOutputImageData;
	float *deviceMaskData;
	float hostMaskData[maskRows * maskColumns] = { 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
			0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
			0.04, 0.04, 0.04, 0.04, 0.04, 0.04, };
	test_images();
	inputImage = PPM_import("computer_programming.ppm");

	assert(maskRows == 5); /* mask height is fixed to 5 in this exercise */
	assert(maskColumns == 5); /* mask width is fixed to 5 in this exercise */

	imageWidth = Image_getWidth(inputImage);
	printf("Image Width %i \n", imageWidth);
	imageHeight = Image_getHeight(inputImage);
	printf("Image Height %i \n", imageHeight);
	imageChannels = Image_getChannels(inputImage);
	printf("Image Channels %i \n", imageChannels);

	outputImage = Image_new(imageWidth, imageHeight, imageChannels);

	hostInputImageData = Image_getData(inputImage);
	hostOutputImageData = Image_getData(outputImage);

	// Allocate device buffers
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
	cudaStatus = cudaMalloc((void**)&deviceInputImageData,imageWidth*imageHeight*imageChannels* sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)&deviceOutputImageData,imageWidth*imageHeight*imageChannels* sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)&deviceMaskData,maskRows*maskColumns* sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	// Copy memory from host to device
	CUDA_CHECK_RETURN(cudaMemcpy(deviceInputImageData, hostInputImageData,sizeof(float) *imageWidth * imageHeight * imageChannels,cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(deviceMaskData, hostMaskData,maskRows*maskColumns* sizeof(float),cudaMemcpyHostToDevice));
	//Grid and Block
	dim3 dimGrid(ceil((float)imageWidth/TILE_WIDTH), ceil((float)imageHeight/TILE_WIDTH));
	dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);
	convolution<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceMaskData,
			deviceOutputImageData, imageChannels, imageWidth, imageHeight);

	// Copy from device to host memory
	CUDA_CHECK_RETURN(cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost));

	PPM_export("processed_computer_programming.ppm", outputImage);

	// Free device memory
	cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceMaskData);

	Image_delete(outputImage);
	Image_delete(inputImage);

	return 0;
}
