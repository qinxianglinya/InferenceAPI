#include "kernel.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include "bboxUtils.h"
#include <cub/cub.cuh>

__global__ void permute(const float* input, float* output, int batchSize, int featureSize, int clsNum)
{
	//tid = 400, featureSize = 400
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= featureSize)
	{
		return;
	}
	//该层每个特征点的总锚框（3个）所预测的数量（置信度+类别置信度+位置）
	int anchorDims = 3 * (4 + 1 + clsNum);
	int numPerbatch = featureSize * anchorDims;

	for (int i = 0; i < batchSize; i++)
	{
		for (int j = 0; j < anchorDims; j++)
		{
			output[i * numPerbatch + tid * anchorDims + j] = input[i * numPerbatch + j * featureSize + tid];
		}
	}
}

__global__ void getDataBySlice(const float* input, float * output, bool forward, int batchSize, int index, int section, int dims)
{
	//index:切片位置
	//section：从某个区间进行切片，section表示区间长度
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	//forward为true，选取从起始位置到切点的数据
	//forward为false，选取从切点位置到末尾的数据
	if (forward)
	{
		for (int i = 0; i < batchSize; i++)
		{
			for (int j = 0; j < index; j++)
			{
				output[i * dims * index + tid * index + j] = input[i * dims * section + tid * section + j];
			}
		}
	}
	else
	{
		for (int i = 0; i < batchSize; i++)
		{
			for (int j = 0; j < (section - index); j++)
			{
				output[i * dims * (section - index) + tid * (section - index) + j] = input[i * dims * section + tid * section + (index + j)];
			}
		}
	}
}

__global__ void sigmoid(const float * input, float* output, int dims)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	output[tid] = exp(input[tid]) / (1 + exp(input[tid]));
}

__global__ void concatTwoTorch(const float* input1, const float* input2, float* output, int dims, int batchSize, int len1, int len2)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	for (int i = 0; i < batchSize; i++)
	{
		for (int j = 0; j < len1; j++)
		{
			output[i * dims * (len1 + len2) + tid * (len1 + len2) + j] = input1[i * dims * len1 + tid * len1 + j];
		}
		for (int j = 0; j < len2; j++)
		{
			output[i * dims * (len1 + len2) + tid * (len1 + len2) + len1 + j] = input2[i * dims * len2 + tid * len2 + j];
		}

	}
}

__global__ void decode(const float* bboxes, const float* pred_bboxes, float* output, int stride, float* x_center, float* y_center,
	float* w, float* h, float* x_center_pred, float* y_center_pred, float* w_pred, float* h_pred, int dims, int batchSize)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}

	int anchor_index = tid % (dims / batchSize);
	x_center[tid] = (bboxes[anchor_index * 4 + 0] + bboxes[anchor_index * 4 + 2]) / 2;
	y_center[tid] = (bboxes[anchor_index * 4 + 1] + bboxes[anchor_index * 4 + 3]) / 2;
	w[tid] = bboxes[anchor_index * 4 + 2] - bboxes[anchor_index * 4 + 0];
	h[tid] = bboxes[anchor_index * 4 + 3] - bboxes[anchor_index * 4 + 1];

	x_center_pred[tid] = (pred_bboxes[tid * 4 + 0] - 0.5) * stride + x_center[tid];
	y_center_pred[tid] = (pred_bboxes[tid * 4 + 1] - 0.5) * stride + y_center[tid];
	w_pred[tid] = exp(pred_bboxes[tid * 4 + 2]) * w[tid];
	h_pred[tid] = exp(pred_bboxes[tid * 4 + 3]) * h[tid];

	output[tid * 4 + 0] = x_center_pred[tid] - w_pred[tid] / 2;
	output[tid * 4 + 1] = y_center_pred[tid] - h_pred[tid] / 2;
	output[tid * 4 + 2] = x_center_pred[tid] + w_pred[tid] / 2;
	output[tid * 4 + 3] = y_center_pred[tid] + h_pred[tid] / 2;

	//x_center[tid] = (bboxes[tid * 4 + 0] + bboxes[tid * 4 + 2]) / 2;
	//y_center[tid] = (bboxes[tid * 4 + 1] + bboxes[tid * 4 + 3]) / 2;
	//w[tid] = bboxes[tid * 4 + 2] - bboxes[tid * 4 + 0];
	//h[tid] = bboxes[tid * 4 + 3] - bboxes[tid * 4 + 1];

	//x_center_pred[tid] = (pred_bboxes[tid * 4 + 0] - 0.5) * stride + x_center[tid];
	//y_center_pred[tid] = (pred_bboxes[tid * 4 + 1] - 0.5) * stride + y_center[tid];
	//w_pred[tid] = exp(pred_bboxes[tid * 4 + 2]) * w[tid];
	//h_pred[tid] = exp(pred_bboxes[tid * 4 + 3]) * h[tid];

	//output[tid * 4 + 0] = x_center_pred[tid] - w_pred[tid] / 2;
	//output[tid * 4 + 1] = y_center_pred[tid] - h_pred[tid] / 2;
	//output[tid * 4 + 2] = x_center_pred[tid] + w_pred[tid] / 2;
	//output[tid * 4 + 3] = y_center_pred[tid] + h_pred[tid] / 2;
}

__global__ void getConfPred(const float* input, float* output, int dims, int clsNum)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	output[tid] = exp(input[tid * (5 + clsNum) + 4]) / (1 + exp(input[tid * (5 + clsNum) + 4]));
}

__global__ void setOffsetYolov3(int* offset, int dims, int batchSize)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid > 0)
	{
		return;
	}
	offset[0] = 0;
	for (int i = 1; i < batchSize + 1; i++)
	{
		offset[i] = i * dims;
	}
}

//用于排序的索引
__global__ void setIndex(int *index, int dims, int batchSize)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	for (int i = 0; i < batchSize; i++)
	{
		index[i*dims + tid] = tid;
	}
}

//divedeLen表示输入的张量的第一个维度，例如bbox，则该值为4， 例如输入为cls_score，则该值为cls_num
//dims表示锚框的数量
__global__ void getTopkData(const float* input, float* output, int dims, int batchSize, int divideLen, int* index, int featureSize)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= 1000)
	{
		return;
	}

	for (int i = 0; i < batchSize; i++)
	{
		int k = index[i * featureSize * 3 + tid];
		for (int j = 0; j < divideLen; j++)
		{
			output[i * 1000 * divideLen + tid * divideLen + j] = input[i * dims * divideLen + k * divideLen + j];
	/*		if (tid == 0 && i == 1)
			{
				printf("%d, %d, %f\n", k, divideLen, input[i * dims * divideLen + k * divideLen + j]);
			}*/
		}
	}
}

//dims = featureSize * 3 * divideLen;
//sum = 三层特征层的topk锚框数量 * divedeLen;
__global__ void concatResult(const float* input, float* output, int batchSize, int dims, int index, int sum)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	for (int i = 0; i < batchSize; i++)
	{
		output[i * sum + index + tid] = input[i * dims + tid];
		//if (i == 1 && index == 0 && tid < 6)
		//{
		//	printf("%d, %f\n", i * sum + index + tid, output[i * sum + index + tid]);
		//}
	}
}

__global__ void test(const float * input, int dims)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	//if (tid == 1200 * 4 + 825 * 4)
	//{
	//	printf("%d:%f\n", tid, input[tid]);
	//}
	if (tid >= 0 && tid < 40)
	{
		printf("%d:%f, %f, %f, %f\n", tid, input[tid * 4 + 0], input[tid * 4 + 1], input[tid * 4 + 2], input[tid * 4 + 3]);
	}
}

__global__ void testInt(const int * input, int dims)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	if (tid >= 1200 && tid < 1200 + 40)
	{
		printf("%d:%d\n", tid, input[tid]);
	}
}

const int TESTBS = 128;
const int TESTGS = (1200 * 8 * 2 + TESTBS - 1) / TESTBS;

//concatIndex表示从该索引处合并结果
extern "C" pluginStatus_t yolov3Detection(cudaStream_t stream, int batchSize, void *workspace, const void* predMap, const void* anchor,
	void* boxOut, void* clsScoreOut, void* confScoreOut, int clsNum, int layerIndex, int featureSize, int stride, int concatIndex, int sum)
{
	const int BS = 128;
	const int GS = (featureSize + BS - 1) / BS;

	void* permutePreMap = workspace;
	size_t permutePreMapSize = floatSize(batchSize, featureSize * 3 * (4 + 1 + clsNum));

	permute << < GS, BS >> > ((float*)predMap, (float*)permutePreMap, batchSize, featureSize, clsNum);


	void* pre_map_conf = nextWorkspacePtr((int8_t*)permutePreMap, permutePreMapSize);
	size_t pre_map_conf_size = floatSize(batchSize, featureSize * 3 * 2);

	const int BS1 = 128;
	const int GS1 = (featureSize * 3 + BS - 1) / BS;
	getDataBySlice << < GS1, BS1 >> > ((float*)permutePreMap, (float*)pre_map_conf, true, batchSize, 2, (5 + clsNum), featureSize * 3);



	void* pre_map_conf_sigmoid = nextWorkspacePtr((int8_t*)pre_map_conf, pre_map_conf_size);
	size_t pre_map_conf_sigmoid_size = floatSize(batchSize, featureSize * 3 * 2);

	const int BS2 = 128;
	const int GS2 = (featureSize * 3 * 2 * batchSize + BS2 - 1) / BS2;
	sigmoid << <GS2, BS2 >> > ((float*)pre_map_conf, (float*)pre_map_conf_sigmoid, (featureSize * 3 * 2 * batchSize));

	void* pred_map_rest = nextWorkspacePtr((int8_t*)pre_map_conf_sigmoid, pre_map_conf_sigmoid_size);
	size_t pred_map_rest_size = floatSize(batchSize, featureSize * 3 * (5 + clsNum - 2));

	const int BS3 = 128;
	const int GS3 = (featureSize * 3 + BS3 - 1) / BS3;
	getDataBySlice << <GS3, BS3 >> > ((float*)permutePreMap, (float*)pred_map_rest, false, batchSize, 2, (5 + clsNum), featureSize * 3);

	const int BS4 = 128;
	const int GS4 = (featureSize * 3 + BS4 - 1) / BS4;
	concatTwoTorch << <GS4, BS4 >> > ((float*)pre_map_conf_sigmoid, (float*)pred_map_rest, (float*)permutePreMap, (featureSize * 3), batchSize, 2, (5 + clsNum - 2));

	void* pred_map_boxes = nextWorkspacePtr((int8_t*)pred_map_rest, pred_map_rest_size);
	size_t pred_map_boxes_size = floatSize(batchSize, featureSize * 3 * 4);

	const int BS5 = 128;
	const int GS5 = (featureSize * 3 + BS5 - 1) / BS5;
	getDataBySlice << <GS5, BS5 >> > ((float*)permutePreMap, (float*)pred_map_boxes, true, batchSize, 4, (5 + clsNum), (featureSize * 3));

	//分配解码所需要的临时空间
	void* x_center = nextWorkspacePtr((int8_t*)pred_map_boxes, pred_map_boxes_size);
	size_t x_center_size = floatSize(batchSize, featureSize * 3);

	void* y_center = nextWorkspacePtr((int8_t*)x_center, x_center_size);
	size_t y_center_size = floatSize(batchSize, featureSize * 3);

	void* w = nextWorkspacePtr((int8_t*)y_center, y_center_size);
	size_t w_size = floatSize(batchSize, featureSize * 3);

	void* h = nextWorkspacePtr((int8_t*)w, w_size);
	size_t h_size = floatSize(batchSize, featureSize * 3);

	void* x_center_pred = nextWorkspacePtr((int8_t*)h, h_size);
	size_t x_center_pred_size = floatSize(batchSize, featureSize * 3);

	void* y_center_pred = nextWorkspacePtr((int8_t*)x_center_pred, x_center_pred_size);
	size_t y_center_pred_size = floatSize(batchSize, featureSize * 3);

	void* w_pred = nextWorkspacePtr((int8_t*)y_center_pred, y_center_pred_size);
	size_t w_pred_size = floatSize(batchSize, featureSize * 3);

	void* h_pred = nextWorkspacePtr((int8_t*)w_pred, w_pred_size);
	size_t h_pred_size = floatSize(batchSize, featureSize * 3);

	void* bbox_pred = nextWorkspacePtr((int8_t*)h_pred, h_pred_size);
	size_t bbox_pred_size = floatSize(batchSize, featureSize * 3 * 4);

	const int BS6 = 128;
	const int GS6 = (featureSize * 3 * batchSize + BS6 - 1) / BS6;
	decode << <GS6, BS6 >> > ((float*)anchor, (float*)pred_map_boxes, (float*)bbox_pred, stride, (float*)x_center,
		(float*)y_center, (float*)w, (float*)h, (float*)x_center_pred, (float*)y_center_pred, (float*)w_pred,
		(float*)h_pred, (batchSize * featureSize * 3), batchSize);

	//test << <TESTGS, TESTBS >> > ((float*)bbox_pred, 1000);
	void* conf_pred = nextWorkspacePtr((int8_t*)bbox_pred, bbox_pred_size);
	size_t conf_pred_size = floatSize(batchSize, featureSize * 3);

	const int BS7 = 128;
	const int GS7 = (featureSize * 3 * batchSize + BS7 - 1) / BS7;
	getConfPred << <GS7, BS7 >> > ((float*)permutePreMap, (float*)conf_pred, (featureSize * 3 * batchSize), clsNum);

	void* cls_pred = nextWorkspacePtr((int8_t*)conf_pred, conf_pred_size);
	size_t cls_pred_size = floatSize(batchSize, featureSize * 3 * (5 + clsNum - 4));

	const int BS8 = 128;
	const int GS8 = (featureSize * 3 + BS8 - 1) / BS8;
	getDataBySlice << <GS8, BS8 >> > ((float*)permutePreMap, (float*)cls_pred, false, batchSize, 5, (5 + clsNum), featureSize * 3);

	void* cls_pred_sigmoid = nextWorkspacePtr((int8_t*)cls_pred, cls_pred_size);
	size_t cls_pred_sigmoid_size = floatSize(batchSize, featureSize * 3 * (5 + clsNum - 4));

	const int BS9 = 128;
	const int GS9 = (batchSize * featureSize * 3 * (5 + clsNum - 4) + BS9 - 1) / BS9;
	sigmoid << <GS9, BS9 >> > ((float*)cls_pred, (float*)cls_pred_sigmoid, batchSize * featureSize * 3 * (5 + clsNum - 4));

	if (featureSize * 3 <= 1000)
	{
		const int BS12 = 128;
		const int GS12 = (featureSize * 3 * 4 + BS12 - 1) / BS12;
		const int GS13 = (featureSize * 3 * 1 + BS12 - 1) / BS12;
		const int GS14 = (featureSize * 3 * clsNum + BS12 - 1) / BS12;
		//bbox
		concatResult << <GS12, BS12 >> > ((float*)bbox_pred, (float*)boxOut, batchSize, featureSize * 3 * 4, concatIndex * 4, sum * 4);
		//conf
		concatResult << <GS13, BS12 >> > ((float*)conf_pred, (float*)confScoreOut, batchSize, featureSize * 3 * 1, concatIndex * 1, sum * 1);
		//cls_score
		concatResult << <GS14, BS12 >> > ((float*)cls_pred_sigmoid, (float*)clsScoreOut, batchSize, featureSize * 3 * clsNum, concatIndex * clsNum, sum * clsNum);
	}
	else
	{
		void     *d_temp_storage = NULL;
		size_t   temp_storage_bytes = 0;
		const int num_items = batchSize * featureSize * 3;
		const int num_segments = batchSize;

		void *d_offsets = nextWorkspacePtr((int8_t*)cls_pred_sigmoid, cls_pred_sigmoid_size);
		size_t offsetSize = (num_segments + 1) * sizeof(int);

		setOffsetYolov3 << <1, 1 >> > ((int*)d_offsets, featureSize * 3, batchSize);
		//d_scoreSorted存放保存排序后的置信度
		void *d_scoreSorted = nextWorkspacePtr((int8_t*)d_offsets, offsetSize);
		size_t scoreSortedSize = floatSize(batchSize, featureSize * 3);
		//indexSorted存放排序后置信度在原始数组中的坐标
		void *indexSorted = nextWorkspacePtr((int8_t*)d_scoreSorted, scoreSortedSize);
		size_t indexSortedSize = intSize(batchSize, featureSize * 3);

		void* index = nextWorkspacePtr((int8_t*)indexSorted, indexSortedSize);
		size_t indexSize = intSize(batchSize, featureSize * 3);

		const int BS10 = 128;
		const int GS10 = (featureSize * 3 + BS10 - 1) / BS10;
		setIndex << <GS10, BS10 >> > ((int*)index, featureSize * 3, batchSize);

		cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, (const float*)conf_pred, (float*)d_scoreSorted, (const int*)index, (int *)indexSorted,
			num_items, num_segments, (const int*)d_offsets, (const int*)d_offsets + 1, 0, sizeof(float) * 8,
			stream);
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
		cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, (const float*)conf_pred, (float*)d_scoreSorted, (const int*)index, (int *)indexSorted,
			num_items, num_segments, (const int*)d_offsets, (const int*)d_offsets + 1, 0, sizeof(float) * 8,
			stream);

		void* bbox_pred_topk = nextWorkspacePtr((int8_t*)index, indexSize);
		size_t bbox_pred_topk_size = floatSize(batchSize, 1000 * 4);

		const int BS11 = 128;
		const int GS11 = (1000 + BS11 - 1) / BS11;
		getTopkData << <GS11, BS11 >> > ((float*)bbox_pred, (float*)bbox_pred_topk, (featureSize * 3), batchSize, 4, (int*)indexSorted, featureSize);

		void* cls_pred_topk = nextWorkspacePtr((int8_t*)bbox_pred_topk, bbox_pred_topk_size);
		size_t cls_pred_topk_size = floatSize(batchSize, 1000 * clsNum);

		getTopkData << <GS11, BS11 >> > ((float*)cls_pred_sigmoid, (float*)cls_pred_topk, (featureSize * 3), batchSize, clsNum, (int*)indexSorted, featureSize);

		void* conf_pred_topk = nextWorkspacePtr((int8_t*)cls_pred_topk, cls_pred_topk_size);
		size_t conf_pred_topk_size = floatSize(batchSize, 1000);

		getTopkData << <GS11, BS11 >> > ((float*)conf_pred, (float*)conf_pred_topk, (featureSize * 3), batchSize, 1, (int*)indexSorted, featureSize);

		const int BS12 = 128;
		const int GS12 = (1000 * 4 + BS12 - 1) / BS12;
		const int GS13 = (1000 * 1 + BS12 - 1) / BS12;
		const int GS14 = (1000 * clsNum + BS12 - 1) / BS12;
		//bbox
		concatResult << <GS12, BS12 >> > ((float*)bbox_pred_topk, (float*)boxOut, batchSize, 1000 * 4, concatIndex * 4, sum * 4);
		//conf
		concatResult << <GS13, BS12 >> > ((float*)conf_pred_topk, (float*)confScoreOut, batchSize, 1000 * 1, concatIndex * 1, sum * 1);
		//cls_score
		concatResult << <GS14, BS12 >> > ((float*)cls_pred_topk, (float*)clsScoreOut, batchSize, 1000 * clsNum, concatIndex * clsNum, sum * clsNum);

		//if (concatIndex == 0)
		//{
		//	test << <TESTGS, TESTBS >> > ((float*)bbox_pred_topk, 1000);
		//	//testInt << <TESTGS, TESTBS >> > ((int*)indexSorted, 1200 * 2);
		//}

		cudaFree(d_temp_storage);
	}


	return STATUS_SUCCESS;

}