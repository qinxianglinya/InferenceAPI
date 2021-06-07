#include "kernel.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>

__global__ void anchorGeneratoryolov3(AnchorParamsYolov3 params, float* output, const int limit)
{
	int width = params.featureSize.nWidth;
	int num_base_anchor = 3;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= limit)
	{
		return;
	}

	const int y = i / (width * num_base_anchor);
	const int x = (i % (width * num_base_anchor)) / num_base_anchor;
	const int base_id = i % num_base_anchor;

	output[i * 4 + 0] = params.fBaseAnchor[base_id].x0 + x * params.nStride;
	output[i * 4 + 1] = params.fBaseAnchor[base_id].y0 + y * params.nStride;
	output[i * 4 + 2] = params.fBaseAnchor[base_id].x1 + x * params.nStride;
	output[i * 4 + 3] = params.fBaseAnchor[base_id].y1 + y * params.nStride;
}


extern "C" pluginStatus_t yolov3Anchor(cudaStream_t stream, AnchorParamsYolov3 params, void *output)
{
	const int dims = params.featureSize.nHeight * params.featureSize.nWidth * 3;

	const int BS = 128;
	const int GS = (dims + BS - 1) / BS;
	anchorGeneratoryolov3 << < GS, BS >> > (params, (float*)output, dims);

	return STATUS_SUCCESS;

}