#include "device_launch_parameters.h"
#include "device_functions.h"
#include "stdio.h"
#include "cuda_runtime.h"  
#include "device_launch_parameters.h" 
#include <math.h>
#include <iostream>
#include <cstdio>

__forceinline__ __device__ float clipp(float in, float low, float high)
{
	return (in < low) ? low : (in > high ? high : in);
}

__global__ void copyKernel(unsigned char* input, unsigned char* output, int index, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size)
	{
		return;
	}
	output[i + index * size * 3] = input[i];
	output[i + size + index * size * 3] = input[i + size];
	output[i + 2 * size + index * size * 3] = input[i + 2 * size];
}

__global__ void copyKernelAlign(unsigned char* input, unsigned char* output, int index, int srcW, int srcH, int dw)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= (srcW * 3 + dw) * srcH)
	{
		return;
	}
	int row = tid / (srcW * 3 + dw);
	int col = tid % (srcW * 3 + dw);
	if (col >= (srcW * 3))
	{
		return;
	}
	output[row * (srcW * 3) + col + index * srcW * srcH * 3] = input[row * (srcW * 3 + dw) + col];
}

__global__ void copyKernelD2(unsigned char* input, unsigned char* output, int index, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size)
	{
		return;
	}
	//printf("%d\n", i);
	//output[i+index*size] = input[i + index * size];
	//output[i + index * size] = input[i];

	output[i + index * size * 3] = input[i];
	output[i + size + index * size * 3] = input[i + size];
	output[i + 2 * size + index * size * 3] = input[i + 2 * size];
	/*if (i == 196607)
	{
		printf("!!!!!!!%d,%f\n", i, (float)input[i]);
	}*/
	//output[i + size + index * size * 3] = input[i + size];
	//output[i + 2 * size + index * size * 3] = input[i + 2 * size];

}

__global__ void resizKernel(unsigned char *inputGpu, float *outputGpu, float* normGpu, int dstW, int dstH, int srcW, int srcH, float mean1, float mean2, float mean3)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k = i % (dstW*dstH);
	int l = i / (dstW*dstH);
	const int x = k % dstW;
	const int y = k / dstW;
	if (x >= dstW || y >= dstH)
		return;
	float ratio_h = float(srcH) / float(dstH);
	float ratio_w = float(srcW) / float(dstW);
	float x0 = float(x) * ratio_w;
	float y0 = float(y) * ratio_h;
	int left = int(clipp((float)floor(x0), 0.0f, float(srcW)));
	int top = int(clipp((float)floor(y0), 0.0f, float(srcH)));
	int right = int(clipp((float)ceil(x0), 0.0f, float(srcW)));
	int bottom = int(clipp((float)ceil(y0), 0.0f, float(srcH)));
	for (int c = 0; c < 3; ++c)
	{
		unsigned char left_top_val = inputGpu[l*srcW*srcH * 3 + top * (srcW * 3) + left * (3) + c];
		unsigned char right_top_val = inputGpu[l*srcW*srcH * 3 + top * (srcW * 3) + right * (3) + c];
		unsigned char left_bottom_val = inputGpu[l*srcW*srcH * 3 + bottom * (srcW * 3) + left * (3) + c];
		unsigned char right_bottom_val = inputGpu[l*srcW*srcH * 3 + bottom * (srcW * 3) + right * (3) + c];
		float top_lerp = left_top_val + (right_top_val - left_top_val) * (x0 - left);
		float bottom_lerp = left_bottom_val + (right_bottom_val - left_bottom_val) * (x0 - left);
		float lerp = clipp((top_lerp + (bottom_lerp - top_lerp) * (y0 - top)), 0.0f, 255.0f);
		outputGpu[i * 3 + c] = lerp;
		//float pixelMean[3]{ 123.68, 116.779, 103.939 };
		if (c == 0)
		{
			normGpu[l*dstW*dstH * 3 + k] = float(outputGpu[i * 3 + c]) - mean1;
		}
		if (c == 1)
		{
			normGpu[l*dstW*dstH * 3 + c * dstW*dstH + k] = float(outputGpu[i * 3 + c]) - mean2;
		}
		if (c == 2)
		{
			normGpu[l*dstW*dstH * 3 + c * dstW*dstH + k] = float(outputGpu[i * 3 + c]) - mean3;
		}

	}
}

__global__ void resizKernel_torch(unsigned char *inputGpu, float *outputGpu, float* normGpu, int dstW, int dstH, int srcW, int srcH, float mean1, float mean2, float mean3)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k = i % (dstW*dstH);
	int l = i / (dstW*dstH);
	const int x = k % dstW;
	const int y = k / dstW;
	if (x >= dstW || y >= dstH)
		return;
	float ratio_h = float(srcH) / float(dstH);
	float ratio_w = float(srcW) / float(dstW);
	float x0 = float(x) * ratio_w;
	float y0 = float(y) * ratio_h;
	int left = int(clipp((float)(x0), 0.0f, float(srcW)));
	int top = int(clipp((float)(y0), 0.0f, float(srcH)));
	int right = int(clipp((float)(x0), 0.0f, float(srcW)));
	int bottom = int(clipp((float)(y0), 0.0f, float(srcH)));
	for (int c = 0; c < 3; ++c)
	{
		unsigned char left_top_val = inputGpu[l*srcW*srcH * 3 + top * (srcW * 3) + left * (3) + c];
		unsigned char right_top_val = inputGpu[l*srcW*srcH * 3 + top * (srcW * 3) + right * (3) + c];
		unsigned char left_bottom_val = inputGpu[l*srcW*srcH * 3 + bottom * (srcW * 3) + left * (3) + c];
		unsigned char right_bottom_val = inputGpu[l*srcW*srcH * 3 + bottom * (srcW * 3) + right * (3) + c];
		float top_lerp = left_top_val + (right_top_val - left_top_val) * (x0 - left);
		float bottom_lerp = left_bottom_val + (right_bottom_val - left_bottom_val) * (x0 - left);
		float lerp = clipp((top_lerp + (bottom_lerp - top_lerp) * (y0 - top)), 0.0f, 255.0f);
		outputGpu[i * 3 + c] = lerp;
		//float pixelMean[3]{ 123.68, 116.779, 103.939 };
		if (c == 0)
		{
			normGpu[l*dstW*dstH * 3 + k] = float(outputGpu[i * 3 + c]) - mean1;
		}
		if (c == 1)
		{
			normGpu[l*dstW*dstH * 3 + c * dstW*dstH + k] = float(outputGpu[i * 3 + c]) - mean2;
		}
		if (c == 2)
		{
			normGpu[l*dstW*dstH * 3 + c * dstW*dstH + k] = float(outputGpu[i * 3 + c]) - mean3;
		}

	}
}


__global__ void paddingKernel(const float* input, float* output, const int resizedW, const int resizedH, const int offset, bool isGreaterWidth)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;


	if (isGreaterWidth)
	{
		int batchIndex = tid / (resizedH * (resizedW + offset));
		int indexInPerImg = tid % (resizedH * (resizedW + offset));
		int x = indexInPerImg % (resizedW + offset); //	列号
		int y = indexInPerImg / (resizedW + offset); // 行号
		if (x >= (resizedW + offset) || y >= resizedH)
			return;

		if (x >= resizedW)
		{
			for (int c = 0; c < 3; c++)
			{
				output[batchIndex * resizedH * (resizedW + offset) * 3 + (y * (resizedW + offset) + x) + c * ((resizedW + offset) *resizedH)] = 0.0f;
			}
		}
		else
		{
			for (int c = 0; c < 3; c++)
			{
				output[batchIndex * resizedH * (resizedW + offset) * 3 + (y * (resizedW + offset) + x) + c * ((resizedW + offset) *resizedH)] = input[batchIndex * resizedH * resizedW * 3 + (y * resizedW + x) + c * (resizedW *
					resizedH)];
			}
		}
	}
	else
	{
		int batchIndex = tid / (resizedW * (resizedH + offset));
		int indexInPerImg = tid % (resizedW * (resizedH + offset));
		int x = indexInPerImg % (resizedW); //	列号
		int y = indexInPerImg / (resizedW); // 行号
		if (x >= (resizedW) || y >= (resizedH + offset))
			return;

		if (y < resizedH)
		{
			for (int c = 0; c < 3; c++)
			{
				output[batchIndex * (resizedH + offset) * resizedW * 3 + (y * resizedW + x) + c * resizedW * (resizedH + offset)] =
					input[batchIndex * resizedH * resizedW * 3 + (y * resizedW + x) + c * resizedW *resizedH];
			}
		}
		else
		{
			for (int c = 0; c < 3; c++)
			{
				output[batchIndex * (resizedH + offset) * resizedW * 3 + (y * resizedW + x) + c * resizedW * (resizedH + offset)] = 0.0f;
			}
		}
	}
}

__global__ void copyArray(float* input, float* output, int dims)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	output[tid] = input[tid];
}

extern "C" void copyImg(void* input, void* output, int index, int k)
{
	const int dim = k;
	const int BS = 512;
	const int GS = (dim + BS - 1) / BS;
	copyKernel << <GS, BS>> > ((unsigned char *)input, (unsigned char *)output, index, dim);
}

extern "C" void copyImgD2(uint8_t* input, uint8_t* output, int index, int k)
{
	const int dim = k;
	const int BS = 512;
	const int GS = (dim + BS - 1) / BS;
	copyKernelD2 << <GS, BS >> > ((unsigned char *)input, (unsigned char *)output, index, dim);
}

extern "C" void copyImgAlign(uint8_t* input, uint8_t* output, int index, int srcW, int srcH, int dw)
{
	const int dims = (srcW * 3 + dw) * srcH;
	const int BS = 512;
	const int GS = (dims + BS - 1) / BS;
	copyKernelAlign << <GS, BS >> > (input, output, index, srcW, srcH, dw);
}


extern "C" void resizeAndNorm(void* inputGpu, void* resizedOutputGpu, void* normGpu, int size, int dstW, int dstH, int srcW, int srcH, float mean1, float mean2, float mean3)
{
	int dim = size;
	const int BS = 128;
	const int GS = (dim + BS - 1) / BS;
	resizKernel << <GS, BS>> > ((unsigned char *)inputGpu, (float *)resizedOutputGpu, (float*)normGpu, dstW, dstH, srcW, srcH, mean1, mean2, mean3);
}


extern "C" void resizeAndNorm_torch(void* inputGpu, void* resizedOutputGpu, void* normGpu, int size, int dstW, int dstH, int srcW, int srcH, float mean1, float mean2, float mean3)
{
	int dim = size;
	const int BS = 128;
	const int GS = (dim + BS - 1) / BS;
	resizKernel_torch << <GS, BS >> > ((unsigned char *)inputGpu, (float *)resizedOutputGpu, (float*)normGpu, dstW, dstH, srcW, srcH, mean1, mean2, mean3);
}


extern "C" void padding(void* input, void* output, int resizedW, int resizedH, int batchSize)
{	//宽大于高

	bool isGreaterWidth;
	int afterPaddingW, afterPaddingH, offset;
	if (resizedW > resizedH)
	{
		isGreaterWidth = true;
		afterPaddingW = (resizedW / 32 + 1) * 32;
		offset = afterPaddingW - resizedW;

		int dims = batchSize * afterPaddingW * resizedH;
		int BS = 128;
		int GS = (dims + BS - 1) / BS;

		paddingKernel << <GS, BS >> > ((float*)input, (float*)output, resizedW, resizedH, offset, isGreaterWidth);
	}
	else if(resizedW < resizedH)
	{
		isGreaterWidth = false;
		afterPaddingH = (resizedH / 32 + 1) * 32;
		offset = afterPaddingH - resizedH;

		int dims = batchSize * afterPaddingH * resizedW;
		int BS = 128;
		int GS = (dims + BS - 1) / BS;

		paddingKernel << <GS, BS >> > ((float*)input, (float*)output, resizedW, resizedH, offset, isGreaterWidth);
	}
	else
	{
		int dims = batchSize * resizedH * resizedW * 3;
		int BS = 128;
		int GS = (dims + BS - 1) / BS;
		copyArray << <GS, BS >> > ((float*)input, (float*)output, dims);
	}
}


static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n", msg, file_name, line_number, cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

//__forceinline__ __device__ float clip(float in, float low, float high)
//{
//	return (in < low) ? low : (in > high ? high-1 : in);
//}

#define clip(x, a, b) x >= a ? (x < b ? x : b-1) : a;

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)


__global__ void resizeD2Kernel(uint8_t* input,
	float* output,
	const int outputWidth,
	const int outputHeight,
	const int inputWidth,
	const int inputHeight,
	const float pixelGroupSizeX,
	const float pixelGroupSizeY,
	const int inputChannels,
	const int batchSizes
	)
{

	//printf("196608:%f\n", (float)input[196608]);
	//printf("196608:%f\n", (float)input[196608]);
	const int dx = blockIdx.x * blockDim.x + threadIdx.x;
	const int dy = blockIdx.y * blockDim.y + threadIdx.y;

	const int pitchInput = inputWidth * inputChannels;
	const int pitchOutput = outputWidth * inputChannels;
	const int inputSize = inputWidth * inputHeight * inputChannels;
	const int outputSize = outputHeight * outputWidth * inputChannels;
	if ((dx < outputWidth) && (dy < outputHeight))
	{
		for (int batchSize = 0; batchSize < batchSizes; batchSize++)
		{
			if (inputChannels == 1) {
			}
			else if (inputChannels == 3) {
				//scale_x:缩放尺寸
				double scale_x = (double)inputWidth / outputWidth;
				double scale_y = (double)inputHeight / outputHeight;

				int xmax = outputWidth;

				float fx = (float)((dx + 0.5) * scale_x - 0.5);
				//floor：向下取整，sx最大为255
				int sx = floor(fx);
				//fx：原始结果与向下取整结果差值
				fx = fx - sx;

				int isx1 = sx;
				if (isx1 < 0) {
					fx = 0.0;
					isx1 = 0;
				}
				if (isx1 >= (inputWidth - 1)) {
					xmax = ::min(xmax, dy);
					fx = 0;
					isx1 = (inputWidth - 1);
				}

				float2 cbufx;
				cbufx.x = (1.f - fx);
				cbufx.y = fx;

				float fy = (float)((dy + 0.5) * scale_y - 0.5);
				int sy = floor(fy);
				fy = fy - sy;

				int isy1 = clip(sy - 1 + 1 + 0, 0, inputHeight);
				int isy2 = clip(sy - 1 + 1 + 1, 0, inputHeight);

				float2 cbufy;
				cbufy.x = (1.f - fy);
				cbufy.y = fy;

				int isx2 = isx1 + 1;
				if (isx2 >= inputWidth)
				{
					isx2 = isx2 - 1;
				}
				float3 d0;

				float3 s11 = make_float3(input[inputSize * (batchSize)+(isy1 * inputWidth + isx1) * inputChannels + 0], input[inputSize * (batchSize)+(isy1 * inputWidth + isx1) * inputChannels + 1], input[inputSize * (batchSize)+(isy1 * inputWidth + isx1) * inputChannels + 2]);
				float3 s12 = make_float3(input[inputSize * (batchSize)+(isy1 * inputWidth + isx2) * inputChannels + 0], input[inputSize * (batchSize)+(isy1 * inputWidth + isx2) * inputChannels + 1], input[inputSize * (batchSize)+(isy1 * inputWidth + isx2) * inputChannels + 2]);
				float3 s21 = make_float3(input[inputSize * (batchSize)+(isy2 * inputWidth + isx1) * inputChannels + 0], input[inputSize * (batchSize)+(isy2 * inputWidth + isx1) * inputChannels + 1], input[inputSize * (batchSize)+(isy2 * inputWidth + isx1) * inputChannels + 2]);
				float3 s22 = make_float3(input[inputSize * (batchSize)+(isy2 * inputWidth + isx2) * inputChannels + 0], input[inputSize * (batchSize)+(isy2 * inputWidth + isx2) * inputChannels + 1], input[inputSize * (batchSize)+(isy2 * inputWidth + isx2) * inputChannels + 2]);

				float h_rst00, h_rst01;

				if (dy > xmax - 1)
				{
					h_rst00 = s11.x;
					h_rst01 = s21.x;
				}
				else
				{
					h_rst00 = s11.x * cbufx.x + s12.x * cbufx.y;
					h_rst01 = s21.x * cbufx.x + s22.x * cbufx.y;
				}

				d0.x = h_rst00 * cbufy.x + h_rst01 * cbufy.y;


				if (dy > xmax - 1)
				{
					h_rst00 = s11.y;
					h_rst01 = s21.y;
				}
				else
				{
					h_rst00 = s11.y * cbufx.x + s12.y * cbufx.y;
					h_rst01 = s21.y * cbufx.x + s22.y * cbufx.y;
				}

				d0.y = h_rst00 * cbufy.x + h_rst01 * cbufy.y;

				if (dy > xmax - 1)
				{
					h_rst00 = s11.z;
					h_rst01 = s21.z;
				}
				else
				{
					h_rst00 = s11.z * cbufx.x + s12.z * cbufx.y;
					h_rst01 = s21.z * cbufx.x + s22.z * cbufx.y;
				}
				d0.z = h_rst00 * cbufy.x + h_rst01 * cbufy.y;

				output[outputSize * (batchSize)+(dy*outputWidth + dx) * 3 + 0] = (d0.x);
				output[outputSize * (batchSize)+(dy*outputWidth + dx) * 3 + 1] = (d0.y);
				output[outputSize * (batchSize)+(dy*outputWidth + dx) * 3 + 2] = (d0.z);
	/*			if ((outputSize * (batchSize)+(dy*outputWidth + dx) * 3 + 1) == (1228798))
				{
					printf("resize:::%f\n", output[outputSize * (batchSize)+(dy*outputWidth + dx) * 3 + 1]);
					printf("s11:%f,%f,%f\n", s11.x, s11.y, s11.z);
					printf("s12:%f,%f,%f\n", s12.x, s12.y, s12.z);
					printf("s21:%f,%f,%f\n", s21.x, s21.y, s21.z);
					printf("s22:%f,%f,%f\n", s22.x, s22.y, s22.z);
					printf("%f,%f,%f,%f\n", h_rst00, cbufy.x, h_rst01, cbufy.y);
					printf("%f,%d,%d, %d,%d\n", (float)input[inputSize * (batchSize)+(isy1 * inputWidth + isx2) * inputChannels + 0],
						(isy1 * inputWidth + isx2) * inputChannels, isy1, isx2, inputSize * (batchSize)+(isy1 * inputWidth + isx2) * inputChannels + 0);
					printf("%d\n", (isy1 * inputWidth + isx1) * inputChannels + 0);
					printf("%d\n", (isy1 * inputWidth + isx2) * inputChannels + 0);
					printf("%d\n", (isy2 * inputWidth + isx1) * inputChannels + 0);
					printf("%d\n", (isy2 * inputWidth + isx2) * inputChannels + 0);
				}*/
			}
			else {

			}
		}
	}
}

__global__ void transformD2Kernel(float* resizedInput, float* transform, int batchSize, int dims)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("transform:%f\n", resizedInput[i]);
	if (i >= (dims * 3))
	{
		return;
	}

	for (int k = 0; k < batchSize; k++)
	{
		if (i % 3 == 0)
		{
			transform[k * 3 * dims + (i / 3)] = resizedInput[k * 3 * dims + i];
		/*	if ((k * 3 * dims + (i / 3)) == (640 * 640 * 2-1))
			{
				printf("!!!!!!!!!!!!test:%f\n", resizedInput[k * 3 * dims + i]);
			}*/
		}
		if (i % 3 == 1)
		{
			transform[k * 3 * dims + dims + (i / 3)] = resizedInput[k * 3 * dims + i];
		/*	if ((k * 3 * dims + dims + (i / 3)) == (640 * 640 * 2-1))
			{
				printf("!!!!!!!!!!!!test:%f\n", resizedInput[k * 3 * dims + i]);
			}*/
		}
		if (i % 3 == 2)
		{
			transform[k * 3 * dims + dims * 2 + (i / 3)] = resizedInput[k * 3 * dims + i];
		/*	if ((k * 3 * dims + dims * 2 + (i / 3)) == (640 * 640 * 2-1))
			{
				printf("!!!!!!!!!!!!test:%f\n", resizedInput[k * 3 * dims + i]);
			}*/
		}
	}



	//for (int c = 0; c < 3; c++)
	//{
	//	transform[batchIndex * (dims / batchSize) * 3 + indexPerImg * 3 + c] = resizedInput[batchIndex * (dims / batchSize) * 3 + c * (dims / batchSize) + indexPerImg];
	//	if (i == 0)
	//	{
	//		//batchIndex * (dims / batchSize) * 3 + c * (dims / batchSize) + indexPerImg
	//		printf("transform:%f\n", resizedInput[batchIndex * (dims / batchSize) * 3 + c * (dims / batchSize) + indexPerImg]);
	//	}
	//}
}

__global__ void normD2Kernel(float* transform, float* norm, int batchSize, int dims, float mean1, float mean2, float mean3, float std1, float std2, float std3)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= (dims*batchSize))
	{
		return;
	}
	int batchIndex = i / (dims);
	int indexPerImg = i % (dims);

	for (int c = 0; c < 3; c++)
	{
		if (c == 0)
		{
			norm[batchIndex * 3 * (dims)+indexPerImg] = ((transform[batchIndex * 3 * (dims)+indexPerImg]) - mean3);
		}
		if (c == 1)
		{
			norm[batchIndex * 3 * (dims)+(dims)+indexPerImg] = (float(transform[batchIndex * 3 * (dims)+(dims)+indexPerImg]) - mean2);
		}
		if (c == 2)
		{
			norm[batchIndex * 3 * (dims)+(dims) * 2 + indexPerImg] = (float(transform[batchIndex * 3 * (dims)+(dims) * 2 + indexPerImg]) - mean1);
		}
	}
}

extern "C" void transformD2(void* resizedInput, void* transform, void* normGpu, int batchSize, int dims, float mean1, float mean2, float mean3, float std1, float std2, float std3)
{
	const int BS = 1024;
	const int GS2 = (dims*batchSize + BS - 1) / BS;
	const int GS1 = (dims * 3 + BS - 1) / BS;
	transformD2Kernel << <GS1, BS >> > ((float*)resizedInput, (float*)transform, batchSize, dims);
	normD2Kernel << <GS2, BS >> > ((float*)transform, (float*)normGpu, batchSize, dims, mean1, mean2, mean3, std1, std2, std3);
}

__global__ void normYolov3Kernel(float* transform, float* norm, int batchSize, int dims, float mean1, float mean2, float mean3, float std1, float std2, float std3)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= (dims*batchSize))
	{
		return;
	}
	int batchIndex = i / (dims);
	int indexPerImg = i % (dims);

	for (int c = 0; c < 3; c++)
	{
		if (c == 0)
		{
			norm[batchIndex * 3 * (dims)+indexPerImg] = (float(transform[batchIndex * 3 * (dims)+indexPerImg]) - mean3) / std3;
		}
		if (c == 1)
		{
			norm[batchIndex * 3 * (dims)+(dims)+indexPerImg] = (float(transform[batchIndex * 3 * (dims)+(dims)+indexPerImg]) - mean2) / std2;
		}
		if (c == 2)
		{
			norm[batchIndex * 3 * (dims)+(dims) * 2 + indexPerImg] = (float(transform[batchIndex * 3 * (dims)+(dims) * 2 + indexPerImg]) - mean1) / std1;
		}
	}
}

extern "C" void transformYolov3(void* resizedInput, void* transform, void* normGpu, int batchSize, int dims, float mean1, float mean2, float mean3, float std1, float std2, float std3)
{
	const int BS = 1024;
	const int GS2 = (dims*batchSize + BS - 1) / BS;
	const int GS1 = (dims * 3 + BS - 1) / BS;
	transformD2Kernel << <GS1, BS >> > ((float*)resizedInput, (float*)transform, batchSize, dims);
	normYolov3Kernel << <GS2, BS >> > ((float*)transform, (float*)normGpu, batchSize, dims, mean1, mean2, mean3, std1, std2, std3);
}




extern "C" void resizeAndNormD2(uint8_t* inputGpu, float* resizedOutputGpu, int dstW, int dstH, int srcW, int srcH, int batchSize)
{
	const dim3 block(16, 16,1);
	//Calculate grid size to cover the whole image
	const dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y,1);
	const dim3 grid1(40, 40, 1);
	const float pixelGroupSizeY = float(srcH) / float(dstH);
	const float pixelGroupSizeX = float(srcW) / float(dstW);
	resizeD2Kernel << <grid, block >> > ((uint8_t*)inputGpu, (float*)resizedOutputGpu, dstW, dstH, srcW, srcH, pixelGroupSizeX, pixelGroupSizeY, 3, batchSize);
}