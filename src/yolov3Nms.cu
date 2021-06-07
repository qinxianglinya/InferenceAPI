#include "kernel.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include "bboxUtils.h"
#include <cub/cub.cuh>

__global__ void boxesScale(const float* input, float* output, int dims, float scale0, float scale1, float scale2, float scale3)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	output[tid * 4] = input[tid * 4] / scale0;
	output[tid * 4 + 1] = input[tid * 4 + 1] / scale1;
	output[tid * 4 + 2] = input[tid * 4 + 2] / scale2;
	output[tid * 4 + 3] = input[tid * 4 + 3] / scale3;
	//printf("%d, %f, %f, %f, %f\n", tid, output[tid * 4], output[tid * 4 + 1], output[tid * 4 + 2], output[tid * 4 + 3]);

}

//dims = sum * (clsNum + 1)
__global__ void paddingScore(const float* input, float* output, int dims, int clsNum, int batchSize, int inputPerbatch)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	for (int i = 0; i < batchSize; i++)
	{
		if (tid % (clsNum + 1) == clsNum)
		{
			//补充0
			output[tid] = 0;
		}
		else
		{
			int k = tid / (clsNum + 1);
			output[i * dims + k * (clsNum + 1) + tid % (clsNum + 1)] = input[i * inputPerbatch + k * clsNum + tid % (clsNum + 1)];
		}
	}
}

//dims等于锚框的总数 * clsNum
__global__ void expandBoxes(const float* input, float* output, int dims, int clsNum)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	int k = tid / clsNum;
	output[tid * 4 + 0] = input[k * 4 + 0];
	output[tid * 4 + 1] = input[k * 4 + 1];
	output[tid * 4 + 2] = input[k * 4 + 2];
	output[tid * 4 + 3] = input[k * 4 + 3];
}

//dims = batchSize * sum * clsNum
__global__ void expandScoreFactors(const float* input, float* output, int dims, int clsNum)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	int k = tid / clsNum;
	output[tid] = input[k];
}

//dims = batchSize * sum * clsNum
__global__ void setLabels(int* output, int dims, int clsNum)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	output[tid] = tid % clsNum;
}

//dims = batchSize * sum * clsNum
__global__ void set_valid_mask(const float* score, float score_thr, int* valid_mask, int dims)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	if (score[tid] > score_thr)
	{
		valid_mask[tid] = 1;
		//printf("%d,%f\n", tid, score[tid]);
	}
	else
	{
		valid_mask[tid] = 0;
	}
}

//dims = batchSize * sum * clsNum
__global__ void resizedClsScore(const float* score, const float* score_factors, float* output, int dims)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	if (score[tid] == (-1))
	{
		output[tid] = -1;
	}
	else
	{
		output[tid] = score[tid] * score_factors[tid];
	/*	if (output[tid] > 0.3)
		{
			printf("%f\n", output[tid]);
		}*/
	}
}

//dims = batchSize * sum * clsNum
__global__ void get_before_nms_data(const float* boxes, const float* scores, const int* labels, const int* index,
	float* boxes_out, float* scores_out, int* labels_out, int dims)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	if (index[tid] == 0)
	{
		boxes_out[tid * 4 + 0] = -1;
		boxes_out[tid * 4 + 1] = -1;
		boxes_out[tid * 4 + 2] = -1;
		boxes_out[tid * 4 + 3] = -1;
		scores_out[tid] = -1;
		labels_out[tid] = -1;
	}
	else
	{
		boxes_out[tid * 4 + 0] = boxes[tid * 4 + 0];
		boxes_out[tid * 4 + 1] = boxes[tid * 4 + 1];
		boxes_out[tid * 4 + 2] = boxes[tid * 4 + 2];
		boxes_out[tid * 4 + 3] = boxes[tid * 4 + 3];
		scores_out[tid] = scores[tid];
		//printf("%d, %f, %f, %f, %f, %f\n", tid, scores_out[tid], boxes_out[tid * 4 + 0], boxes_out[tid * 4 + 1], boxes_out[tid * 4 + 2], boxes_out[tid * 4 + 3]);
		labels_out[tid] = labels[tid];
	}

}

//设置排序时所需的offset
__global__ void setOffsetBox_Score(int *offset, int dims, int batchSize)
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
	/*offset[tid] = 0;
	offset[tid + 1] = dims;*/
}

//求nms_boxes时所需的box偏移量
__global__ void getOffsetBox(const int* clsIndex, const float* max_coordinate, float* offset, int dims, int batchSize, const float* before_nms_boxes)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	int numPerbatch = dims;
	for (int i = 0; i < batchSize; i++)
	{
		if (before_nms_boxes[i * dims * 4 + tid * 4] == (-1))
		{
			offset[i * numPerbatch + tid] = 0;
		}
		else
		{
			offset[i * numPerbatch + tid] = clsIndex[i * numPerbatch + tid] * (max_coordinate[i * dims * 4] + 1);
			//printf("%f\n", max_coordinate[i * dims * 4]);

			//printf("%d, %d, %f\n", tid, clsIndex[i * numPerbatch + tid], offset[i * numPerbatch + tid]);
		}
	}
}

__global__ void get_boxes_for_nms(const float* boxes_before_nms, const float* offset, float* boxes_for_nms, int dims)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	if (boxes_before_nms[tid * 4 + 0] == (-1) && boxes_before_nms[tid * 4 + 1] == (-1) &&
		boxes_before_nms[tid * 4 + 2] == (-1) && boxes_before_nms[tid * 4 + 3] == (-1))
	{
		boxes_for_nms[tid * 4 + 0] = (-1);
		boxes_for_nms[tid * 4 + 1] = (-1);
		boxes_for_nms[tid * 4 + 2] = (-1);
		boxes_for_nms[tid * 4 + 3] = (-1);
	}
	else
	{
		boxes_for_nms[tid * 4 + 0] = boxes_before_nms[tid * 4 + 0] + offset[tid];
		boxes_for_nms[tid * 4 + 1] = boxes_before_nms[tid * 4 + 1] + offset[tid];
		boxes_for_nms[tid * 4 + 2] = boxes_before_nms[tid * 4 + 2] + offset[tid];
		boxes_for_nms[tid * 4 + 3] = boxes_before_nms[tid * 4 + 3] + offset[tid];
		//printf("%d, %f,%f,%f,%f\n", tid, boxes_for_nms[tid * 4 + 0], boxes_for_nms[tid * 4 + 1], boxes_for_nms[tid * 4 + 2], boxes_for_nms[tid * 4 + 3]);
		//printf("%d,%f,%f\n", tid, boxes_before_nms[tid * 4 + 0], offset[tid]);
		//if(tid >= (dims / 2))
		//printf("offset:%f\n", offset[tid]);
	}
}

__global__ void setIndexYolov3(int* input, int dims, int batchSize)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	for (int i = 0; i < batchSize; i++)
	{
		input[i * dims + tid] = tid;
	}
}

__device__ float bboxSizeYolov3(
	const float xmin, const float ymin, const float xmax, const float ymax)
{
	//printf("box size called...\n");
	if (xmax < xmin || ymax < ymin)
	{
		// If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
		return 0;
	}
	else
	{
		float width = xmax - xmin;
		float height = ymax - ymin;

		return width * height;
	}
}

#define max(a,b) ( ((a)>(b)) ? (a):(b) )
#define min(a,b) ( ((a)>(b)) ? (b):(a) )

__device__ float intersectBboxYolov3(
	const float xmin1, const float ymin1, const float xmax1, const float ymax1,
	const float xmin2, const float ymin2, const float xmax2, const float ymax2)
{
	if (xmin2 > xmax1 || xmax2 < xmin1 || ymin2 > ymax1 || ymax2 < ymin1)
	{
		// Return [0, 0, 0, 0] if there is no intersection.
		return 0;
	}
	else
	{
		return bboxSizeYolov3(max(xmin1, xmin2), max(ymin1, ymin2),
			min(xmax1, xmax2), min(ymax1, ymax2));
	}
}


__device__ float getIouYolov3(const float xmin1, const float ymin1, const float xmax1, const float ymax1,
	const float xmin2, const float ymin2, const float xmax2, const float ymax2)
{
	float intersect = intersectBboxYolov3(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2);

	if (intersect > 0)
	{
		float bbox1_size = bboxSizeYolov3(xmin1, ymin1, xmax1, ymax1);
		float bbox2_size = bboxSizeYolov3(xmin2, ymin2, xmax2, ymax2);
		return intersect / (bbox1_size + bbox2_size - intersect);
	}
	else
	{
		return 0.;
	}
}

__global__ void setSuppressed(int* suppressed, int dims)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	suppressed[tid] = 0;
}

__global__  void nmsYolov3(const float* box_before_nms, const float* predictLoc, const float* before_nms_scores, int* before_nms_labels,
	float* scores_after_nms,  int * labels_after_nms, const int* index, int *suppressed, int* keep, float* dets, float iouThreshold, const int batchSize, const int numPerbatch)
{
	//dims= batchsize
	//tid:0->(k-1)
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= batchSize)
	{
		return;
	}
	/**
	1、得分最高框，计算其面积
	2、计算其他框与其iou大小，大于阈值则弃用
	3、

	*/
	int sizePerbatch = numPerbatch;
	int sum = 0;
	for (int j = 0; j < numPerbatch; j++)
	{
		int s = index[sizePerbatch * tid + j];

		if (suppressed[sizePerbatch * tid + s] == 1)
		{
			continue;
		}
		/*if(box_before_nms[sizePerbatch * tid * 4 + s * 4] != (-1))
		printf("%f,%f,%f,%f,%f\n", box_before_nms[sizePerbatch * tid * 4 + s * 4], box_before_nms[sizePerbatch * tid * 4 + s * 4 + 1],
			box_before_nms[sizePerbatch * tid * 4 + s * 4 + 2], box_before_nms[sizePerbatch * tid * 4 + s * 4 + 3],before_nms_scores[sizePerbatch * tid + s]);*/

		//keep[100 * tid + sum] = s;
		dets[100 * 4 * tid + sum * 4 + 0] = box_before_nms[sizePerbatch * tid * 4 + s * 4];
		dets[100 * 4 * tid + sum * 4 + 1] = box_before_nms[sizePerbatch * tid * 4 + s * 4 + 1];
		dets[100 * 4 * tid + sum * 4 + 2] = box_before_nms[sizePerbatch * tid * 4 + s * 4 + 2];
		dets[100 * 4 * tid + sum * 4 + 3] = box_before_nms[sizePerbatch * tid * 4 + s * 4 + 3];
		scores_after_nms[100 * tid + sum] = before_nms_scores[sizePerbatch * tid + s];
		labels_after_nms[100 * tid + sum] = before_nms_labels[sizePerbatch * tid + s];
		//printf("%f, %f, %f, %f, %f\n", dets[100 * 4 * tid + sum * 4 + 0], dets[100 * 4 * tid + sum * 4 + 1],
		//	dets[100 * 4 * tid + sum * 4 + 2], dets[100 * 4 * tid + sum * 4 + 3], scores_after_nms[100 * tid + sum]);
		sum++;
		if (sum == 100)
		{
			break;
		}
		//printf("%f, %f, %f, %f, %f\n", dets[100 * 4 * tid + sum * 4 + 0], dets[100 * 4 * tid + sum * 4 + 1],
		//	dets[100 * 4 * tid + sum * 4 + 2], dets[100 * 4 * tid + sum * 4 + 3], scores_after_nms[100 * tid + sum]);
		//if(tid == 1)
		//printf("%f, %f, %f, %f, %f\n", box_before_nms[sizePerbatch * tid * 4 + s * 4], box_before_nms[sizePerbatch * tid * 4 + s * 4 + 1],
		//	box_before_nms[sizePerbatch * tid * 4 + s * 4 + 2], box_before_nms[sizePerbatch * tid * 4 + s * 4 + 3], before_nms_scores[sizePerbatch * tid + s]);

		//printf("%f\n", before_nms_scores[sizePerbatch * tid + s]);


		//计算面积
		float xmin = predictLoc[sizePerbatch * tid * 4 + s * 4];
		float ymin = predictLoc[sizePerbatch * tid * 4 + s * 4 + 1];
		float xmax = predictLoc[sizePerbatch * tid * 4 + s * 4 + 2];
		float ymax = predictLoc[sizePerbatch * tid * 4 + s * 4 + 3];

		for (int k = j + 1; k < numPerbatch; k++)
		{
			//比较iou,大于则弃用
			int d = index[sizePerbatch * tid + k];
			float xmin1 = predictLoc[sizePerbatch * tid * 4 + d * 4];
			float ymin1 = predictLoc[sizePerbatch * tid * 4 + d * 4 + 1];
			float xmax1 = predictLoc[sizePerbatch * tid * 4 + d * 4 + 2];
			float ymax1 = predictLoc[sizePerbatch * tid * 4 + d * 4 + 3];
			float size2 = bboxSizeYolov3(xmin1, ymin1, xmax1, ymax1);

			if (size2 <= 0)
			{
				suppressed[sizePerbatch * tid + d] = 1;
				//continue;
			}

			float iou;
			iou = getIouYolov3(xmin, ymin, xmax, ymax, xmin1, ymin1, xmax1, ymax1);
			if (iou > iouThreshold)
			{
				//printf("%f\n", before_nms_scores[sizePerbatch * tid + d]);
				suppressed[sizePerbatch * tid + d] = 1;
			}
		}
	}
}

__global__ void get_conf_inds(const float* mlvl_conf, const float conf_thr, int* conf_inds, int dims)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	if (mlvl_conf[tid] >= conf_thr)
	{
		conf_inds[tid] = 1;
		//printf("%d, %f\n", tid, mlvl_conf[tid]);
	}
	else
	{
		conf_inds[tid] = -1;
	}
}

__global__ void get_positive_data(const float* all_box, const float* all_scores, const float* all_conf, const int* conf_inds,
	float* positive_box, float* positive_scores, float* positive_conf, int dims, int clsNum)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	if (conf_inds[tid] != (-1))
	{
		positive_box[tid * 4 + 0] = all_box[tid * 4 + 0];
		positive_box[tid * 4 + 1] = all_box[tid * 4 + 1];
		positive_box[tid * 4 + 2] = all_box[tid * 4 + 2];
		positive_box[tid * 4 + 3] = all_box[tid * 4 + 3];
		for (int i = 0; i < clsNum; i++)
		{
			positive_scores[tid * clsNum + i] = all_scores[tid * clsNum + i];
		}
		/*positive_scores[tid] = all_scores[tid];*/
		positive_conf[tid] = all_conf[tid];
	}
	else
	{
		positive_box[tid * 4 + 0] = 0;
		positive_box[tid * 4 + 1] = 0;
		positive_box[tid * 4 + 2] = 0;
		positive_box[tid * 4 + 3] = 0;
		for (int i = 0; i < clsNum; i++)
		{
			positive_scores[tid * clsNum + i] = (-1);
		}
		//positive_scores[tid] = (-1);
		positive_conf[tid] = (-1);
	}
}

__global__ void initResult(float* box, float* score, int* labels, int batchSize)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= 100 * batchSize)
	{
		return;
	}
	box[tid * 4 + 0] = (-1.0);
	box[tid * 4 + 1] = (-1.0);
	box[tid * 4 + 2] = (-1.0);
	box[tid * 4 + 3] = (-1.0);
	score[tid] = (-1.0);
	labels[tid] = -1;
}

__global__ void returnResult(const float* box, const float* score, const int* label, float* box_out, float* score_out, int* label_out, float score_thr, const int dims)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	if (score[tid] < score_thr)
	{
		score_out[tid] = 0;
		box_out[tid * 4 + 0] = -1;
		box_out[tid * 4 + 1] = -1;
		box_out[tid * 4 + 2] = -1;
		box_out[tid * 4 + 3] = -1;
		label_out[tid] = -1;
	}
	else
	{
		score_out[tid] = score[tid];
		box_out[tid * 4 + 0] = box[tid * 4 + 0];
		box_out[tid * 4 + 1] = box[tid * 4 + 1];
		box_out[tid * 4 + 2] = box[tid * 4 + 2];
		box_out[tid * 4 + 3] = box[tid * 4 + 3];
		//printf("%f, %f, %f, %f, %f\n", box_out[tid * 4 + 0], box_out[tid * 4 + 1], box_out[tid * 4 + 2], box_out[tid * 4 + 3], score_out[tid]);
		label_out[tid] = label[tid];
	}
}


__global__ void test12(const float * input, const float* input2 , int dims)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= 1)
	{
		return;
	}
	for (int i = 0; i < 3000 * 3; i++)
	{
		if (input[i * 4] != (-1))
		{
			printf("%f, %f, %f, %f, %f\n", input[i * 4], input[i * 4 + 1], input[i * 4 + 2], input[i * 4 + 3], input2[i]);
		}
	}

	//printf("%d:%f\n", tid, input[tid]);
	//if (tid >= (dims /2) && tid <=(dims/2+40))
	//if(tid <=30)
	//{
	//	//printf("test\n");
	//	//if(input[tid * 4] != -1)
	//	printf("%d: %f, %f, \n", tid, input[tid]);

	//	//printf("%d: %f, %f, %f, %f\n", tid, input[tid * 4 + 0], input[tid * 4 + 1], input[tid * 4 + 2], input[tid * 4 + 3]);
	//}
}

__global__ void testInt1(const int * input, int dims)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	int sum;
	for (int i = 0; i < 3000 * 4; i++)
	{
		if (input[i] == 0)
		{
			sum++;
		}
	}
	printf("sum:%d\n, ", sum);
}

const int TESTBS = 128;
const int TESTGS = (3000 * 3 * 4 + TESTBS - 1) / TESTBS;
extern "C" pluginStatus_t yolov3Nms(cudaStream_t stream, int batchSize, void *workspace, const float* mlvl_bboxes, const float* mlvl_scores,
	const float* mlvl_conf_scores, float conf_thr, float score_thr, float boxScale0, float boxScale1, float boxScale2, float boxScale3, int sum, int clsNum, float* det_bboxes, int* det_labels, float* det_scores, float iouThreshold)
{
	//printf("test\n");
	const int BS1 = 128;
	const int GS1 = (batchSize * sum + BS1 - 1) / BS1;

	void* scaled_boxes = workspace;
	size_t scaled_boxes_size = floatSize(batchSize, sum * 4);

	boxesScale << <GS1, BS1 >> > ((float*)mlvl_bboxes, (float*)scaled_boxes, (batchSize * sum), boxScale0, boxScale1, boxScale2, boxScale3);

	void* conf_index = nextWorkspacePtr((int8_t*)scaled_boxes, scaled_boxes_size);
	size_t conf_index_size = intSize(batchSize, sum);

	const int BS_1 = 128;
	const int GS_1 = (batchSize * sum + BS_1 - 1) / BS_1;
	get_conf_inds << <GS_1, BS_1 >> > ((float*)mlvl_conf_scores, conf_thr, (int*)conf_index, batchSize * sum);
	//testInt1 << <TESTGS, TESTBS >> > ((int*)conf_index, sum);
	//test12 << <GS_1, BS_1 >> > ((float*)mlvl_conf_scores, sum);

	void* positive_bboxes = nextWorkspacePtr((int8_t*)conf_index, conf_index_size);
	size_t positive_bboxes_size = floatSize(batchSize, sum * 4);
	void* positive_scores = nextWorkspacePtr((int8_t*)positive_bboxes, positive_bboxes_size);
	size_t positive_scores_size = floatSize(batchSize, sum * clsNum);
	void* positive_conf = nextWorkspacePtr((int8_t*)positive_scores, positive_scores_size);
	size_t positive_conf_size = floatSize(batchSize, sum);

	const int BS_2 = 128;
	const int GS_2 = (batchSize * sum + BS_2 - 1) / BS_2;
	get_positive_data << <GS_2, BS_2 >> > ((float*)scaled_boxes, (float*)mlvl_scores, (float*)mlvl_conf_scores, (int*)conf_index,
		(float*)positive_bboxes, (float*)positive_scores, (float*)positive_conf, sum * batchSize, clsNum);

	//test12 << <GS_1, BS_1 >> > ((float*)positive_bboxes, sum);

	void* bboxes = nextWorkspacePtr((int8_t*)positive_conf, positive_conf_size);
	size_t bboxes_size = floatSize(batchSize, sum * clsNum * 4);

	const int BS2 = 128;
	const int GS2 = (batchSize * sum * clsNum + BS2 - 1) / BS2;
	expandBoxes << <GS2, BS2 >> > ((float*)positive_bboxes, (float*)bboxes, batchSize * sum * clsNum, clsNum);

	void* labels = nextWorkspacePtr((int8_t*)bboxes, bboxes_size);
	size_t labels_size = intSize(batchSize, sum * clsNum);

	const int BS3 = 128;
	const int GS3 = (batchSize * sum * clsNum + BS3 - 1) / BS3;
	setLabels << <GS3, BS3 >> > ((int*)labels, batchSize * sum * clsNum, clsNum);

	void* valid_mask = nextWorkspacePtr((int8_t*)labels, labels_size);
	size_t valid_mask_size = intSize(batchSize, sum * 3);

	const int BS4 = 128;
	const int GS4 = (batchSize * sum * clsNum + BS4 - 1) / BS4;
	set_valid_mask << <GS4, BS4 >> > ((float*)positive_scores, score_thr, (int*)valid_mask, batchSize * sum * clsNum);

	void* score_factors = nextWorkspacePtr((int8_t*)valid_mask, valid_mask_size);
	size_t score_factors_size = floatSize(batchSize, sum * clsNum);

	const int BS5 = 128;
	const int GS5 = (batchSize * sum * clsNum + BS5 - 1) / BS5;
	expandScoreFactors << <GS5, BS5 >> > ((float*)positive_conf, (float*)score_factors, batchSize * sum * clsNum, clsNum);

	void* resized_scores = nextWorkspacePtr((int8_t*)score_factors, score_factors_size);
	size_t resized_scores_size = floatSize(batchSize, sum * clsNum);

	const int BS6 = 128;
	const int GS6 = (batchSize * sum * clsNum + BS6 -1) / BS6;
	resizedClsScore << <GS6, BS6 >> > ((float*)positive_scores, (float*)score_factors, (float*)resized_scores, batchSize * sum * clsNum);

	//test1 << <TESTGS, TESTBS >> > ((float*)positive_conf, sum);

	void* before_nms_boxes = nextWorkspacePtr((int8_t*)resized_scores, resized_scores_size);
	size_t before_nms_boxes_size = intSize(batchSize, sum * clsNum * 4);

	void* before_nms_scores = nextWorkspacePtr((int8_t*)before_nms_boxes, before_nms_boxes_size);
	size_t before_nms_scores_size = floatSize(batchSize, sum * clsNum);

	void* before_nms_labels = nextWorkspacePtr((int8_t*)before_nms_scores, before_nms_scores_size);
	size_t before_nms_labels_size = intSize(batchSize, sum * clsNum);

	const int BS7 = 128;
	const int GS7 = (batchSize * sum * clsNum + BS7 - 1) / BS7;
	
	get_before_nms_data << <GS7, BS7 >> > ((float*)bboxes, (float*)resized_scores, (int*)labels, (int*)valid_mask, (float*)before_nms_boxes,
		(float*)before_nms_scores, (int*)before_nms_labels, batchSize * sum * clsNum);



	//排序，找出box最大值
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;
	int num_segments = batchSize;

	void *sortedBox = nextWorkspacePtr((int8_t*)before_nms_labels, before_nms_labels_size);
	size_t sortedBox_size = floatSize(batchSize, sum * clsNum * 4);

	void *offsets = nextWorkspacePtr((int8_t*)sortedBox, sortedBox_size);
	size_t offsetSize = (num_segments + 1) * sizeof(int);
	//setoffset用于排序用
	setOffsetBox_Score << <1, 1 >> > ((int*)offsets, sum * clsNum * 4, batchSize);

	cub::DeviceSegmentedRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes, (const float*)before_nms_boxes, (float*)sortedBox, batchSize * sum * clsNum * 4,
		num_segments, (const int*)offsets, (const int*)offsets + 1, 0, sizeof(float) * 8, stream);
	// Allocate temporary storage
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	// Run sorting operation
	cub::DeviceSegmentedRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes, (const float*)before_nms_boxes, (float*)sortedBox, batchSize * sum * clsNum * 4,
		num_segments, (const int*)offsets, (const int*)offsets + 1, 0, sizeof(float) * 8, stream);

	void* offset_for_boxes = nextWorkspacePtr((int8_t*)offsets, offsetSize);
	size_t offset_for_boxes_size = floatSize(batchSize, sum * clsNum);


	const int BS8 = 128;
	const int GS8 = (sum * clsNum + BS8 - 1) / BS8;
	getOffsetBox<<<GS8, BS8>>>((int*)before_nms_labels, (float*)sortedBox, (float*)offset_for_boxes, sum * clsNum, batchSize, (float*)before_nms_boxes);

	

	void* boxes_for_nms = nextWorkspacePtr((int8_t*)offset_for_boxes, offset_for_boxes_size);
	size_t boxes_for_nms_size = floatSize(batchSize, sum * clsNum * 4);

	const int BS9 = 128;
	const int GS9 = (batchSize * sum * clsNum * 4 + BS9 - 1) / BS9;
	get_boxes_for_nms<<<GS9, BS9>>>((float*)before_nms_boxes, (float*)offset_for_boxes, (float*)boxes_for_nms, batchSize * sum * clsNum);
	
	
	//test12 << <TESTGS, TESTBS >> > ((float*)boxes_for_nms,(float*)before_nms_scores,  1);


	//正式进行nms处理
	//1、先对所有分数进行排序
	void* offsetScore = nextWorkspacePtr((int8_t*)boxes_for_nms, boxes_for_nms_size);
	size_t offsetScoreSize = (num_segments + 1) * sizeof(int);
	setOffsetBox_Score << <1, 1 >> > ((int*)offsetScore, sum * clsNum, batchSize);


	void *sortedScore = nextWorkspacePtr((int8_t*)offsetScore, offsetScoreSize);
	size_t sortedScoreSize = sigmoidDataSize(batchSize, sum * clsNum);

	void* indexPtr = nextWorkspacePtr((int8_t*)sortedScore, sortedScoreSize);
	size_t indexPtr_size = intSize(batchSize, sum * clsNum);

	const int BS10 = 128;
	const int GS10 = (sum * clsNum + BS10 - 1) / BS10;
	setIndexYolov3 << <GS10, BS10 >> > ((int*)indexPtr, sum * clsNum, batchSize);

	void *indexSorted = nextWorkspacePtr((int8_t*)indexPtr, indexPtr_size);
	size_t indexSorted_size = intSize(batchSize, sum * clsNum);

	cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, (const float*)before_nms_scores, (float*)sortedScore, (const int*)indexPtr, (int *)indexSorted,
		batchSize * sum * clsNum, num_segments, (const int*)offsetScore, (const int*)offsetScore + 1, 0, sizeof(float) * 8, stream);
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, (const float*)before_nms_scores, (float*)sortedScore, (const int*)indexPtr, (int *)indexSorted,
		batchSize * sum * clsNum, num_segments, (const int*)offsetScore, (const int*)offsetScore + 1, 0, sizeof(float) * 8, stream);

	//test12 << <GS10, BS10 >> > ((float*)sortedScore, 30);

	void *suppressBox = nextWorkspacePtr((int8_t*)indexSorted, indexSorted_size);
	size_t suppressBoxSize = intSize(batchSize, sum * clsNum);
	void* keep = nextWorkspacePtr((int8_t*)suppressBox, suppressBoxSize);
	size_t keepSize = intSize(batchSize, 100);
	void* dets = nextWorkspacePtr((int8_t*)keep, keepSize);
	size_t dets_size = floatSize(batchSize, 100 * 4);
	void* scores_after_nms = nextWorkspacePtr((int8_t*)dets, dets_size);
	size_t scores_after_nms_size = floatSize(batchSize, 100);
	void* labels_after_nms = nextWorkspacePtr((int8_t*)scores_after_nms, scores_after_nms_size);
	size_t labels_after_nms_size = intSize(batchSize, 100);

	const int BS10_1 = 128;
	const int GS10_1 = (batchSize * 1000 + BS10_1 - 1) / BS10_1;
	initResult << <GS10_1, BS10_1 >> > ((float*)dets, (float*)scores_after_nms, (int*)labels_after_nms, batchSize);


	const int BS11 = 128;
	const int GS11 = (batchSize * sum * clsNum + BS11 - 1) / BS11;
	setSuppressed << <GS11, BS11 >> > ((int*)suppressBox, batchSize * sum * clsNum);

	//test12 << <GS_1, BS_1 >> > ((float*)positive_bboxes, sum);

	nmsYolov3 << <40, 128 >> > ((const float*)before_nms_boxes, (float*)boxes_for_nms, (float*)before_nms_scores, (int*)before_nms_labels, (float*)scores_after_nms,
		(int*)labels_after_nms, (int*)indexSorted, (int*)suppressBox, (int*)keep, (float*)dets, iouThreshold, batchSize, sum * clsNum);

	//test1 << <TESTGS, TESTBS >> > ((float*)dets, batchSize * 100);
	//testInt1 << <TESTGS, TESTBS >> > ((int*)suppressBox, 1);

	const int BS12 = 128;
	const int GS12 = (100 * batchSize + BS12 - 1) / BS12;
	returnResult << <GS12, BS12 >> > ((float*)dets, (float*)scores_after_nms, (int*)labels_after_nms, (float*)det_bboxes, (float*)det_scores, (int*)det_labels, score_thr, 100 * batchSize);



	cudaFree(d_temp_storage);
	return STATUS_SUCCESS;
}