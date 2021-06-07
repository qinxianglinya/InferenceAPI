/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "kernel.h"
#include "reducedMath.h"
#include <iostream>
#include <math.h>
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>


using nvinfer1::rt::reduced_divisor;
//template <unsigned nthdsPerCTA>
//__launch_bounds__(nthdsPerCTA)
__global__ void gridAnchorKernel(
	const GridAnchorParametersTf param,
	const int numAspectRatios,
	//reduced_divisor divObj,
	const float* widths,
	const float* heights,
	float* outputData
)
{
	// output dims: (H, W, param.numMinSize, (1+haveMaxSize+numAR-1), 4)
	 //dim:产生锚框的数量
	const int dim = param.H * param.W * numAspectRatios;

	/*
	 * Parameters used to calculate the bounding box coordinates back to input image scale
	 * Normally we calculate the anchorStride = image_input_size (in pixel) / feature_map_size
	 * Here we do not use image_input_size for the moment
	 * Instead we use 1.0
	 * The coordinates calculated are scaled by the input image size.
	 * Most of the coordinates will be in a range of [0, 1], except for the bounding box coordinates going outside of the image
	 * Every coordinate will go back to the pixel coordinates in the input image if being multiplied by image_input_size
	 * Here we implicitly assumes the image input and feature map are square
	 */
	 // float anchorStrideH = (1.0 / param.H);
	 // float anchorStrideW = (1.0 / param.W);
	 // float anchorOffsetH = 0.5 * anchorStrideH;
	 // float anchorOffsetW = 0.5 * anchorStrideW;
	 //anchorStride:根据level进行计算
	float anchorStrideH = pow(2.0, param.level);
	float anchorStrideW = pow(2.0, param.level);
	float anchorOffsetH = 0;
	float anchorOffsetW = 0;
	if ((param.imgH % (int)anchorStrideH) == 0 || (param.imgH == 1))
	{
		 anchorOffsetH = 0.5 * anchorStrideH;
	}
	if ((param.imgW % (int)anchorStrideH) == 0 || (param.imgW == 1))
	{
		anchorOffsetW = 0.5 * anchorStrideW;
	}
	
	
	//tid：数据索引
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dim)
	{
		return;
	}

	int arId, currIndex;
	//divObj.divmod(tid, currIndex, arId);
	currIndex = tid / 6;
	arId = tid - (6 * currIndex);
	//w:当前线程生成锚框所在的位置
	//h:当前线程生成锚框所在的位置
	const int w = currIndex % param.W;
	const int h = currIndex / param.W;

	// Center coordinates
	float yC = h * anchorStrideH + anchorOffsetH;
	float xC = w * anchorStrideW + anchorOffsetW;

	// x_min, y_min 
	float xMin = xC - 0.5 * widths[arId];
	float yMin = yC - 0.5 * heights[arId];

	// x_max, y_max 
	float xMax = xC + 0.5 * widths[arId];
	float yMax = yC + 0.5 * heights[arId];

	//normalize_coordinates坐标归一化
	float a = xMin / param.imgW;
	float b = yMin / param.imgH;

	float c = xMax / param.imgW;
	float d = yMax / param.imgH;

	outputData[tid * 4] = a;
	outputData[tid * 4 + 1] = b;
	outputData[tid * 4 + 2] = c;
	outputData[tid * 4 + 3] = d;

	// Remember to move the output cursor
	float* output = outputData + dim * 4;

	// Simply copying the variance
	output[tid * 4] = param.variance[0];
	output[tid * 4 + 1] = param.variance[1];
	output[tid * 4 + 2] = param.variance[2];
	output[tid * 4 + 3] = param.variance[3];
	//if (tid == 0)
	//{
	//	printf("-----------------\n");
	//	printf("%f\n", a);
	//	printf("%f\n", b);
	//	printf("%f\n", c);
	//	printf("%f\n", d);
	//	//printf("%d\n", w);
	//	//printf("%d\n", h);
	//	printf("xc:%f\n", xC);
	//	printf("yc:%f\n", yC);
	//	printf("%f\n", widths[0]);
	//	printf("%f\n", heights[0]);
	//	printf("variance1:%f\n", param.variance[0]);
	//	printf("variance2:%f\n", param.variance[1]);
	//	printf("variance3:%f\n", param.variance[2]);
	//	printf("variance4:%f\n", param.variance[3]);
	//	printf("level:%d\n", param.level);
	//	printf("feature map w:%d\n", param.W);
	//	printf("feature map h:%d\n", param.H);
	//	printf("image w:%d\n", param.imgW);
	//	printf("image h:%d\n", param.imgH);
	//	printf("anchorStrideH:%f\n", anchorOffsetH);
	//	printf("anchorStrideW:%f\n", anchorOffsetW);
	//	printf("-----------------\n");
	//}
}


//widths,height:锚框的宽高
//numAspectRatios:单位像素生成的锚框数量

extern "C" pluginStatus_t anchorGridInference(
	cudaStream_t stream,
	const GridAnchorParametersTf param,
	const int numAspectRatios,
	const void* widths,
	const void* heights,
	void* outputData
)
{
	//dim:19x19x3=1083
	const int dim = param.H * param.W * numAspectRatios;
	//reduced_divisor divObj(numAspectRatios);
	if (dim > 5120)
	{
		//printf("grid block1\n");
		//BS：一个block中线程数
		//GS：每个Grid中block数
		const int BS = 128;
		const int GS = (dim + BS - 1) / BS;
		gridAnchorKernel<<< GS, BS, 0, stream >>> (param, numAspectRatios, 
			(const float*)widths, (const float*)heights,
			(float*)outputData);

	}
	else
	{
		//printf("grid block2\n");
		const int BS = 32;
		const int GS = (dim + BS - 1) / BS;
		gridAnchorKernel<<< GS, BS, 0, stream >>> (param, numAspectRatios, 
			(const float*)widths, (const float*)heights,
			(float*)outputData);
	}
	CSC(cudaGetLastError(), STATUS_FAILURE);
	return STATUS_SUCCESS;
}
