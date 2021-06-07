#include "kernel.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <iostream>
#include "anchorGeneratorTorch.h"

__global__ void anchorGenerator(AnchorParamsTorch params, float *output)
{
	const int dim = params.featureSize.nHeight * params.featureSize.nWidth * 9;
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dim)
	{
		return;
	}
	

	//currentIndex:表示锚框所在位置的序号
	//arId:表示锚框在当前位置的排序（一个位置存在9个锚框）
	int currentIndex, arId;
	currentIndex = tid / 9;
	arId = tid - (currentIndex * 9);

	int col = currentIndex % params.featureSize.nWidth;//列号
	int row = currentIndex / params.featureSize.nWidth;//行号

	float x0 = params.fBaseAnchor[arId].x0 + col * params.nStride;
	float y0 = params.fBaseAnchor[arId].y0 + row * params.nStride;
	float x1 = params.fBaseAnchor[arId].x1 + col * params.nStride;
	float y1 = params.fBaseAnchor[arId].y1 + row * params.nStride;

	output[tid * 4] = x0;
	output[tid * 4 + 1] = y0;
	output[tid * 4 + 2] = x1;
	output[tid * 4 + 3] = y1;
	//if (tid < 3)
	//{
	//	printf("x0= %f\n", x0);
	//	printf("y0= %f\n", y0);
	//	printf("x1= %f\n", x1);
	//	printf("y1= %f\n", y1);
	//	printf("*********************\n");
	//}
}


extern "C" pluginStatus_t anchorGridTorch(cudaStream_t stream, AnchorParamsTorch params, void *output)
{
	const int dims = params.featureSize.nHeight * params.featureSize.nWidth * 9;
	//std::cout << "dims" << dims << std::endl;
	const int BS = 128;
	const int GS = (dims + BS - 1) / BS;
	anchorGenerator << < GS, BS>> > (params, (float*)output);

	return STATUS_SUCCESS;
}