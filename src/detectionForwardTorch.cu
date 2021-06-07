#include <cub/cub.cuh>
#include "kernel.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <cuda_runtime.h>
#include "bboxUtils.h"


__global__ void sigmoid(const float *confData, float *confSigmoid, int *indexPtr, int dim, int batchSize)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dim)
	{
		return;
	}
	for (int i = 0; i < batchSize; i++)
	{
		confSigmoid[i*dim + tid] = exp(confData[i*dim + tid]) / (1 + exp(confData[i*dim + tid]));
		indexPtr[i*dim + tid] = tid;
	}
}

__device__ float bboxSize(
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

__device__ float intersectBbox(
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
		return bboxSize(max(xmin1, xmin2), max(ymin1, ymin2),
			min(xmax1, xmax2), min(ymax1, ymax2));
	}
}


__device__ float getIou(const float xmin1, const float ymin1, const float xmax1, const float ymax1,
	const float xmin2, const float ymin2, const float xmax2, const float ymax2)
{
	float intersect = intersectBbox(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2);

	if (intersect > 0)
	{
		float bbox1_size = bboxSize(xmin1, ymin1, xmax1, ymax1);
		float bbox2_size = bboxSize(xmin2, ymin2, xmax2, ymax2);
		return intersect / (bbox1_size + bbox2_size - intersect);
	}
	else
	{
		return 0.;
	}
}

//suppressed:0表示保留，1表示弃用
__global__  void nms(const float* predictLoc, const int* index,
	bool *suppressed, float iouThreshold, const int dims, int nbLayer, int keepTopK, int batchSize)
{
	//dims：从0到sizePerbatch；
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}

	int sizePerbatch = nbLayer * keepTopK;

	for (int i = 0; i < batchSize; i++)
	{
		int s = index[i * sizePerbatch + tid];

		//计算面积
		float xmin = predictLoc[sizePerbatch * i * 4 + s * 4];
		float ymin = predictLoc[sizePerbatch * i * 4 + s * 4 + 1];
		float xmax = predictLoc[sizePerbatch * i * 4 + s * 4 + 2];
		float ymax = predictLoc[sizePerbatch * i * 4 + s * 4 + 3];

		if ((tid + 1) < sizePerbatch)
		{
			for (int j = tid + 1; j < sizePerbatch; j++)
			{
				//比较iou,大于则弃用
				int d = index[i * sizePerbatch + j];
				float xmin1 = predictLoc[sizePerbatch * i * 4 + d * 4];
				float ymin1 = predictLoc[sizePerbatch * i * 4 + d * 4 + 1];
				float xmax1 = predictLoc[sizePerbatch * i * 4 + d * 4 + 2];
				float ymax1 = predictLoc[sizePerbatch * i * 4 + d * 4 + 3];
				float size2 = bboxSize(xmin1, ymin1, xmax1, ymax1);

				if (size2 <= 0)
				{
					suppressed[i * sizePerbatch * sizePerbatch + tid * sizePerbatch + j] = true;
				}
				else
				{
					float iou;
					iou = getIou(xmin, ymin, xmax, ymax, xmin1, ymin1, xmax1, ymax1);
					if (iou > iouThreshold)
					{
						suppressed[i * sizePerbatch * sizePerbatch + tid * sizePerbatch + j] = true;
					}
				}
			}
		}
	}
}




//priorNum:每个网格先验框的个数
__global__ void permuteData(const float *input, float *output, int num, int devideNum, int featureSize, int priorNum, int batchSize)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= num)
	{
		return;
	}
	int numPerbatch = num * devideNum * priorNum;

	//printf("before permute score: %f\n", input[tid]);
	//if (tid == 0)
	//{
	//	
	//}

	for (int s = 0; s < batchSize; s++)
	{
		for (int i = 0; i < priorNum; i++)
		{
			for (int j = 0; j < devideNum; j++)
			{
				output[s*numPerbatch + tid * priorNum*devideNum + i * devideNum + j] = input[s*numPerbatch + (i * devideNum*featureSize) + (j*featureSize) + tid];
				/*	if (tid == 0 && s == 1)
					{
						if (i == 0  && devideNum == 5)
						{
							printf("conf input%d:%f\n", tid, input[tid]);
							printf("input:%f\n", input[s*numPerbatch + (i * devideNum*featureSize) + (j*featureSize) + tid]);
							printf("output:%f\n", output[s*numPerbatch + tid * priorNum*devideNum + i * devideNum + j]);
						}
					}*/
			}
		}
	}

}

__global__ void permuteData2(const float *input, float *output, int num, int devideNum, int featureSize, int priorNum, int batchSize)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= num)
	{
		return;
	}
	int numPerbatch = num * devideNum * priorNum;

	//if (tid == 0)
	//{
	//	printf("before permute score: %f\n", input[tid]);
	//}
	//for (int s = 0; s < batchSize; s++)
	//{
	//	if (s == 1&&tid==0)
	//	{
	//		printf("%f\n", input[s*numPerbatch]);
	//	}
	//}

	for (int s = 0; s < batchSize; s++)
	{
		for (int i = 0; i < priorNum; i++)
		{
			for (int j = 0; j < devideNum; j++)
			{
				output[s*numPerbatch + tid * priorNum*devideNum + i * devideNum + j] = input[s*numPerbatch + (i * devideNum*featureSize) + (j*featureSize) + tid];
				/*			if (s == 1 && i == 0)
							{
								printf("output:%f\n", output[s*numPerbatch + tid * priorNum*devideNum + i * devideNum + j]);
							}*/
							/*	if (tid == 0 && s == 1)
								{
									if (i == 0  && devideNum == 5)
									{
										printf("conf input%d:%f\n", tid, input[tid]);
										printf("input:%f\n", input[s*numPerbatch + (i * devideNum*featureSize) + (j*featureSize) + tid]);
										printf("output:%f\n", output[s*numPerbatch + tid * priorNum*devideNum + i * devideNum + j]);
									}
								}*/
			}
		}
	}

}


__global__ void setOffset(int *offset, int dims, int batchSize)
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

__global__ void getTopkNum(const float *inputScore, const int *inputIndex, float *outputScore, int *outputIndex,
	float threshold, const int dims, int *anchorIndex, int *classIndex, const int classNum, int batchSize, int totalScoreNum)
{
	//dims为keeptopk
	//totalScoreNum:featureSize * 9 * numCls
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	//printf("inputIndex:%d\n", inputIndex[tid]);
	//outputScore[tid] = inputScore[tid];
	//outputIndex[tid] = inputIndex[tid];
	for (int i = 0; i < batchSize; i++)
	{
		if (inputScore[i*totalScoreNum + tid] >= threshold)
		{
			//printf("%f\n", inputScore[i*totalScoreNum + tid]);
			outputScore[i*dims + tid] = inputScore[i*totalScoreNum + tid];
			outputIndex[i*dims + tid] = inputIndex[i*totalScoreNum + tid];
			//upThreshold[tid] = 1;
			anchorIndex[i*dims + tid] = outputIndex[i*dims + tid] / classNum;//锚框索引

			//printf("anchorindex:%d\n", anchorIndex[tid]);
			classIndex[i*dims + tid] = outputIndex[i*dims + tid] % classNum;//类别编号
			//if(i==1)
			//printf("%d\n", anchorIndex[i*dims + tid]);
	/*		if (i == 1)
			{
				printf("%d\n", anchorIndex[i*dims + tid]);
			}*/
		}
		else
		{
			//upThreshold[tid] = 0;
			outputScore[i*dims + tid] = 0.0f;
			outputIndex[i*dims + tid] = -1;
			anchorIndex[i*dims + tid] = -1;
			classIndex[i*dims + tid] = -1;
		}
		/*	if (i == 1)
			{
				printf("anchorIndex[i*dims + tid]:%d\n", anchorIndex[i*dims + tid]);
			}*/
	}


}

//将每一层的输出结果进行连接
__global__ void concatArray(const float* beforeBox, const float* beforeScore, const int* beforeClass, float* afterBox, float* afterScore, int* afterClass,
	int layerIndex, int dims, int batchSize, int keepK, int layerNum, int keepTopK)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	for (int i = 0; i < batchSize; i++)
	{
		if (tid < keepK)
		{
			//融合box
			afterBox[i*dims * 4 * layerNum + layerIndex * 4 * keepTopK + tid * 4] = beforeBox[i*keepK * 4 + tid * 4];
			afterBox[i*dims * 4 * layerNum + layerIndex * 4 * keepTopK + tid * 4 + 1] = beforeBox[i*keepK * 4 + tid * 4 + 1];
			afterBox[i*dims * 4 * layerNum + layerIndex * 4 * keepTopK + tid * 4 + 2] = beforeBox[i*keepK * 4 + tid * 4 + 2];
			afterBox[i*dims * 4 * layerNum + layerIndex * 4 * keepTopK + tid * 4 + 3] = beforeBox[i*keepK * 4 + tid * 4 + 3];
			//融合score
			afterScore[i* layerNum * keepTopK + layerIndex * keepTopK + tid] = beforeScore[i*keepK + tid];
			//if (afterScore[i* layerNum * keepTopK + layerIndex * keepTopK + tid] != 0)
			//{
			//	printf("concat score:!!!%f\n", afterScore[i* layerNum * keepTopK + layerIndex * keepTopK + tid]);
			//}
			//融合class
			afterClass[i* layerNum * keepTopK + layerIndex * keepTopK + tid] = beforeClass[i*keepK + tid];
		}
		else
		{
			//融合box
			//printf("tid con:%d\n", tid);
			afterBox[i*dims * 4 * layerNum + layerIndex * 4 * keepTopK + tid * 4] = 0;
			afterBox[i*dims * 4 * layerNum + layerIndex * 4 * keepTopK + tid * 4 + 1] = 0;
			afterBox[i*dims * 4 * layerNum + layerIndex * 4 * keepTopK + tid * 4 + 2] = 0;
			afterBox[i*dims * 4 * layerNum + layerIndex * 4 * keepTopK + tid * 4 + 3] = 0;
			//融合score
			afterScore[i*layerNum * keepTopK + layerIndex * keepTopK + tid] = 0;
			//融合class
			afterClass[i*layerNum * keepTopK + layerIndex * keepTopK + tid] = (-1);
		}

		//if (i == 1 && afterScore[i*layerNum * keepTopK + layerIndex * keepTopK + tid] != 0&&layerIndex == 2)
		//{
		//	//printf("layer index:%d\n", layerIndex);
		//	printf("after box:%d, %f, %f, %f, %f, %f, %d\n", tid, afterBox[i*dims * 4 * layerNum + layerIndex * 4 * keepTopK + tid * 4],
		//		afterBox[i*dims * 4 * layerNum + layerIndex * 4 * keepTopK + tid * 4 + 1], afterBox[i*dims * 4 * layerNum + layerIndex * 4 * keepTopK + tid * 4 + 2],
		//		afterBox[i*dims * 4 * layerNum + layerIndex * 4 * keepTopK + tid * 4 + 3], afterScore[i*layerNum * keepTopK + layerIndex * keepTopK + tid],
		//		afterClass[i*layerNum * keepTopK + layerIndex * keepTopK + tid]);
		//	printf("before box:%d, %f, %f, %f, %f, %f, %d\n", tid, beforeBox[i*keepK * 4 + tid * 4], beforeBox[i*keepK * 4 + tid * 4 + 1], beforeBox[i*keepK * 4 + tid * 4 + 2],
		//		beforeBox[i*keepK * 4 + tid * 4 + 3], beforeScore[i*keepK + tid], beforeClass[i*keepK + tid]);
		//}

	}
}

__global__ void decode(const float *anchor, const float *locData, float *predictBox, int dims, float scaleClamp, int batchSize)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	if (locData[tid] != 0)
	{

	}
	for (int i = 0; i < batchSize; i++)
	{
		//if (anchor[i*dims * 4 +tid * 4] == 0.0f&&locData[i*dims * 4 + tid * 4] == 0.0f&&
		//	anchor[i*dims * 4 + tid * 4+1] == 0.0f&&locData[i*dims * 4 + tid * 4+1] == 0.0f&&
		//	anchor[i*dims * 4 + tid * 4+2] == 0.0f&&locData[i*dims * 4 + tid * 4+2] == 0.0f&&
		//	anchor[i*dims * 4 + tid * 4+3] == 0.0f&&locData[i*dims * 4 + tid * 4+3] == 0.0f)
		//{
		//	predictBox[i*dims*4 + tid * 4] = 0.0f;
		//	predictBox[i*dims*4 + tid * 4 + 1] = 0.0f;
		//	predictBox[i*dims*4 + tid * 4 + 2] = 0.0f;
		//	predictBox[i*dims*4 + tid * 4 + 3] = 0.0f;
		//	return;
		//}

		//进行解码操作
		//torch.clamp:限制最大最小值
		float anchorW = anchor[i*dims * 4 + tid * 4 + 2] - anchor[i*dims * 4 + tid * 4];
		float anchorH = anchor[i*dims * 4 + tid * 4 + 3] - anchor[i*dims * 4 + tid * 4 + 1];
		float anchorCx = anchor[i*dims * 4 + tid * 4] + 0.5 * anchorW;
		float anchorCy = anchor[i*dims * 4 + tid * 4 + 1] + 0.5 * anchorH;

		float dx = locData[i*dims * 4 + tid * 4];
		float dy = locData[i*dims * 4 + tid * 4 + 1];
		float dw = locData[i*dims * 4 + tid * 4 + 2];
		float dh = locData[i*dims * 4 + tid * 4 + 3];
		/*	if (locData[tid] != 0)
			{
				printf("decode input:%f, %f, %f, %f\n", locData[tid * 4], locData[tid * 4 + 1], locData[tid * 4 + 1], locData[tid * 4 + 1]);
				printf("%f, %f, %f, %f, %f, %f, %f, %f\n", anchorW, anchorH, anchorCx, anchorCy, dx, dy, dw, dh);

			}*/

		if (dw > scaleClamp)
		{
			dw = scaleClamp;
		}
		if (dh > scaleClamp)
		{
			dh = scaleClamp;
		}

		float preCx = dx * anchorW + anchorCx;
		float preCy = dy * anchorH + anchorCy;
		float preW = anchorW * exp(dw);
		float preH = anchorH * exp(dh);

		predictBox[i*dims * 4 + tid * 4] = preCx - 0.5 * preW;
		predictBox[i*dims * 4 + tid * 4 + 1] = preCy - 0.5 * preH;
		predictBox[i*dims * 4 + tid * 4 + 2] = preCx + 0.5 * preW;
		predictBox[i*dims * 4 + tid * 4 + 3] = preCy + 0.5 * preH;
		/*	if (i == 1&& predictBox[i*dims * 4 + tid * 4]!=0)
			{
				printf("predict box: %d,  %f, %f,  %f,  %f  \n", i*dims * 4 + tid * 4, predictBox[i*dims * 4 + tid * 4], predictBox[i*dims * 4 + tid * 4 + 1],
					predictBox[i*dims * 4 + tid * 4 + 2], predictBox[i*dims * 4 + tid * 4 + 3]);
			}*/
	}


}

__global__ void getBoxAndAnchor(const int *anchorIndex, float *inputAnchor, float *inputLoc, float *outputAnchor, float *outputLoc, int dims, int batchSize, int boxStart)
{
	//dims : keeptopk
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}

	for (int i = 0; i < batchSize; i++)
	{
		//if (i == 1 && tid < 10)
		//{
		//	printf("%f\n", inputLoc[tid]);
		//}
		if (anchorIndex[i*dims + tid] == (-1))
		{
			outputAnchor[i*dims * 4 + tid * 4] = 0.0f;
			outputAnchor[i*dims * 4 + tid * 4 + 1] = 0.0f;
			outputAnchor[i*dims * 4 + tid * 4 + 2] = 0.0f;
			outputAnchor[i*dims * 4 + tid * 4 + 3] = 0.0f;
			outputLoc[i*dims * 4 + tid * 4] = 0.0f;
			outputLoc[i*dims * 4 + tid * 4 + 1] = 0.0f;
			outputLoc[i*dims * 4 + tid * 4 + 2] = 0.0f;
			outputLoc[i*dims * 4 + tid * 4 + 3] = 0.0f;
		}
		else
		{
			//printf("%d\n", (anchorIndex[tid]));
			outputAnchor[i*dims * 4 + tid * 4] = inputAnchor[(anchorIndex[i*dims + tid]) * 4];
			outputAnchor[i*dims * 4 + tid * 4 + 1] = inputAnchor[(anchorIndex[i*dims + tid]) * 4 + 1];
			outputAnchor[i*dims * 4 + tid * 4 + 2] = inputAnchor[(anchorIndex[i*dims + tid]) * 4 + 2];
			outputAnchor[i*dims * 4 + tid * 4 + 3] = inputAnchor[(anchorIndex[i*dims + tid]) * 4 + 3];



			outputLoc[i*dims * 4 + tid * 4] = inputLoc[boxStart*i + (anchorIndex[i*dims + tid]) * 4];
			outputLoc[i*dims * 4 + tid * 4 + 1] = inputLoc[boxStart*i + (anchorIndex[i*dims + tid]) * 4 + 1];
			outputLoc[i*dims * 4 + tid * 4 + 2] = inputLoc[boxStart*i + (anchorIndex[i*dims + tid]) * 4 + 2];
			outputLoc[i*dims * 4 + tid * 4 + 3] = inputLoc[boxStart*i + (anchorIndex[i*dims + tid]) * 4 + 3];

			//if (i == 1)
			//{
			///*	printf("%f,%f,%f,%f\n", outputLoc[i*dims * 4 + tid * 4],
			//		outputLoc[i*dims * 4 + tid * 4 + 1],
			//		outputLoc[i*dims * 4 + tid * 4 + 2],
			//		outputLoc[i*dims * 4 + tid * 4 + 3]);*/
			//	printf("%d, %f\n", anchorIndex[i*dims + tid], inputLoc[(anchorIndex[i*dims + tid]) * 4]);
			//}

		/*	if (tid < 10)
			{
				printf("input anchor:%f, %f, %f, %f\n", inputAnchor[tid * 4], inputAnchor[tid * 4 + 1], inputAnchor[tid * 4 + 2], inputAnchor[tid * 4 + 3]);
				printf("outputAnchor:%f, %f, %f, %f\n", outputAnchor[i*dims * 4 + tid * 4], outputAnchor[i*dims * 4 + tid * 4 + 1],
					outputAnchor[i*dims * 4 + tid * 4 + 2], outputAnchor[i*dims * 4 + tid * 4 + 3]);
			}*/
			//printf("outputAnchor:%f, %f, %f, %f\n", outputAnchor[i*dims * 4 + tid * 4], outputAnchor[i*dims * 4 + tid * 4 + 1],
			//	outputAnchor[i*dims * 4 + tid * 4 + 2], outputAnchor[i*dims * 4 + tid * 4 + 3]);

			/*printf("output loc%d,%f\n", anchorIndex[tid], outputLoc[i*dims * 4 + tid * 4]);
			printf("output loc%d,%f\n", anchorIndex[tid], outputLoc[i*dims * 4 + tid * 4 + 1]);
			printf("output loc%d,%f\n", anchorIndex[tid], outputLoc[i*dims * 4 + tid * 4 + 2]);
			printf("output loc%d,%f\n", anchorIndex[tid], outputLoc[i*dims * 4 + tid * 4 + 3]);*/
		}
	}

}


__global__ void getBoxForNms(const float* box, const int* classIndex, const float* sortedBox, float* boxForNms, int batchSize, int layerNum, int dims, int* indexPtr, int* suppressBox, int keepTopK)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	//printf("getboxfornms\n");
	for (int i = 0; i < batchSize; i++)
	{
		float maxbox = sortedBox[i * keepTopK * 4 * layerNum] + 1;
		/*	if (tid == 0)
			{
				printf("maxbox:%f\n", maxbox);
			}*/

		indexPtr[i * keepTopK * layerNum + tid] = tid;
		suppressBox[i * keepTopK * layerNum + tid] = 0;
		//float offset = maxbox * 
	/*	if (tid == 2002)
		{
			printf("2002:%f, %f, %f, %f\n", box[2002 * 4], box[2002 * 4 + 1], box[2002 * 4 + 2], box[2002 * 4 + 3]);
		}*/
		if (box[i * keepTopK * layerNum * 4 + tid * 4] == 0 && box[i * keepTopK * layerNum * 4 + tid * 4 + 3] == 0)
		{
			boxForNms[i * keepTopK * layerNum * 4 + tid * 4] = 0.0f;
			boxForNms[i * keepTopK * layerNum * 4 + tid * 4 + 1] = 0.0f;
			boxForNms[i * keepTopK * layerNum * 4 + tid * 4 + 2] = 0.0f;
			boxForNms[i * keepTopK * layerNum * 4 + tid * 4 + 3] = 0.0f;
		}
		else
		{
			//printf("%d\n", classIndex[2002]);
			//printf("class index%d:%d\n", i * keepTopK * layerNum + tid,classIndex[i * keepTopK * layerNum + tid]);

			//printf("maxbox:%d\n", classIndex[i * keepTopK * layerNum + tid]);
			boxForNms[i * keepTopK * layerNum * 4 + tid * 4] = box[i * keepTopK * layerNum * 4 + tid * 4] + maxbox * classIndex[i * keepTopK * layerNum + tid];
			boxForNms[i * keepTopK * layerNum * 4 + tid * 4 + 1] = box[i * keepTopK * layerNum * 4 + tid * 4 + 1] + maxbox * classIndex[i * keepTopK * layerNum + tid];
			boxForNms[i * keepTopK * layerNum * 4 + tid * 4 + 2] = box[i * keepTopK * layerNum * 4 + tid * 4 + 2] + maxbox * classIndex[i * keepTopK * layerNum + tid];
			boxForNms[i * keepTopK * layerNum * 4 + tid * 4 + 3] = box[i * keepTopK * layerNum * 4 + tid * 4 + 3] + maxbox * classIndex[i * keepTopK * layerNum + tid];
			//debug 0118 -- boxforNms 
			//offset：类别索引*当前box最大值作为offset
		/*	printf("%f,%f,%f,%f\n", boxForNms[i * keepTopK * layerNum * 4 + tid * 4],
				boxForNms[i * keepTopK * layerNum * 4 + tid * 4 + 1],
				boxForNms[i * keepTopK * layerNum * 4 + tid * 4 + 2],
				boxForNms[i * keepTopK * layerNum * 4 + tid * 4 + 3]);*/
		}
	}
}

__global__ void getResultAfterNms(const float* box, const float* score, const int* classIndex, const int* indexSorted, const bool* suppressedIndex, float* scoreAfterNms,
	float* boxAfterNms, int* classIndexAfterNms, int topK, int batchSize, int layerNum, int keepTopK)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	//if(tid>=topK)
	if (tid >= layerNum * keepTopK)
	{
		return;
	}
	int numPerBatch = layerNum * keepTopK;
	for (int i = 0; i < batchSize; i++)
	{
		//printf("tid:%d\n", tid);
		int k = indexSorted[i*numPerBatch + tid];
		if (suppressedIndex[i*numPerBatch + tid] == false)
		{
			boxAfterNms[i*numPerBatch * 4 + tid * 4] = box[i*numPerBatch * 4 + k * 4];
			boxAfterNms[i*numPerBatch * 4 + tid * 4 + 1] = box[i*numPerBatch * 4 + k * 4 + 1];
			boxAfterNms[i*numPerBatch * 4 + tid * 4 + 2] = box[i*numPerBatch * 4 + k * 4 + 2];
			boxAfterNms[i*numPerBatch * 4 + tid * 4 + 3] = box[i*numPerBatch * 4 + k * 4 + 3];
			scoreAfterNms[i*numPerBatch + tid] = score[i*numPerBatch + k];
			classIndexAfterNms[i*numPerBatch + tid] = classIndex[i*numPerBatch + k];//待确认
		}
		else
		{
			//int k = indexSorted[i*numPerBatch + tid];
			boxAfterNms[i*numPerBatch * 4 + tid * 4] = 0.0f;
			boxAfterNms[i*numPerBatch * 4 + tid * 4 + 1] = 0.0f;
			boxAfterNms[i*numPerBatch * 4 + tid * 4 + 2] = 0.0f;
			boxAfterNms[i*numPerBatch * 4 + tid * 4 + 3] = 0.0f;
			scoreAfterNms[i*numPerBatch + tid] = 0.0f;
			//printf("getresult2:%f\n", scoreAfterNms[i*numPerBatch + tid]);
			classIndexAfterNms[i*numPerBatch + tid] = -1;
		}
	}

}
//
__device__ float clamp(float data, int limitMax)
{
	if (data < 0)
	{
		return 0.0f;
	}
	else if (data > limitMax)
	{
		return limitMax * 1.0;
	}
	else
	{
		return data;
	}
}

//__global__ void scaleAndClip(const float* box, float* boxAfterScale, int srcW, int srcH, float scaleW, float scaleH, int topK, int batchSize, int layerNum, int keepTopK)
//{
//	int tid = blockIdx.x * blockDim.x + threadIdx.x;
//	topK = layerNum * keepTopK;
//	if (tid >= topK)
//	{
//		return;
//	}
//	for (int i = 0; i < batchSize; i++)
//	{
//		//if (box[i* topK * 4 + tid * 4] != 0)
//		//{
//		//	//printf("%f, %f, %f, %f\n", box[i* topK * 4 + tid * 4], box[i* topK * 4 + tid * 4 + 1], box[i* topK * 4 + tid * 4 + 2], box[i* topK * 4 + tid * 4 + 3]);
//		//}
//		if (box[i* topK * 4 + tid*4] != 0 && box[i* topK * 4 + tid * 4 + 1] != 0 &&
//			box[i* topK * 4 + tid*4 + 2] != 0 && box[i* topK * 4 + tid*4 + 3] != 0)
//		{
//			float xmin = box[i* topK * 4 + tid*4] * scaleW;
//			float xmax = box[i*topK * 4 + tid * 4 + 2] * scaleW;
//			float ymin = box[i*topK * 4 + tid * 4 + 1] * scaleH;
//			float ymax = box[i*topK * 4 + tid * 4 + 3] * scaleH;
//			//printf("%f, %f, %f, %f\n", xmin, ymin, xmax, ymax);
//			xmin = clamp(xmin, srcW);
//			xmax = clamp(xmax, srcW);
//			ymin = clamp(ymin, srcH);
//			ymax = clamp(ymax, srcH);
//			/*printf("%f, %f, %f, %f\n", xmin, ymin, xmax, ymax);*/
//
//			boxAfterScale[i * topK * 4 + tid * 4] = xmin;
//			boxAfterScale[i * topK * 4 + tid * 4 + 1] = ymin;
//			boxAfterScale[i * topK * 4 + tid * 4 + 2] = xmax;
//			boxAfterScale[i * topK * 4 + tid * 4 + 3] = ymax;
//		}
//		else
//		{
//			boxAfterScale[i * topK * 4 + tid * 4] = 0.0f;
//			boxAfterScale[i * topK * 4 + tid * 4 + 1] = 0.0f;
//			boxAfterScale[i * topK * 4 + tid * 4 + 2] = 0.0f;
//			boxAfterScale[i * topK * 4 + tid * 4 + 3] = 0.0f;
//		}
//	}
//
//
//}

//debug permute
__global__ void test(float* input, const int dims)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	if (tid == 0)
	{
		printf("%f\n", input[tid]);
	}
	//if (tid == 6163 * 4)
	//{
	//	printf("%f\n", input[6163 * 4]);
	//}
	//if (tid >= (dims / 2)&&tid<=(dims/2)+100)
	//{
	//	printf("%d, %f\n", tid, input[tid]);
	//		//input[tid*4], input[tid*4+1], input[tid*4+2], input[tid*4+3]);
	//}
}
//debug index
__global__ void test1(float* input, int dims)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	if (input[tid * 4] != 0)
	{
		printf("%f, %f, %f, %f\n", input[tid * 4], input[tid * 4 + 1], input[tid * 4 + 2], input[tid * 4 + 3]);
	}
	//if (tid >= (dims/2) && tid < (dims / 2)+30)
	//{
	//	printf("%d\n", input[tid]);
	//}
}

void* detectionInferenceTorch(cudaStream_t stream, int batchSize, void *workspace, const void *anchor, const void *confData, const void *locData, void *boxPtr,
	void *scorePtr, void *classPtr, int layerIndex, int featureSize, int priorNum, int classNum, int keepK, int layerNum, float scoreThreshold, int keepTopK)
{


	const int dims = featureSize * priorNum * classNum;
	const int BS = 128;
	const int GS = (dims + BS - 1) / BS;

	//1、改变预测置信度的维度
	void *permuteConf = workspace;
	size_t permuteConfSize = floatSize(batchSize, featureSize * priorNum * classNum);

	const int GS1 = (featureSize + BS - 1) / BS;
	permuteData << <GS1, BS >> > ((float*)confData, (float*)permuteConf, dims / priorNum / classNum, classNum, featureSize, priorNum, batchSize);
	//std::cout << "--------------------------------" << std::endl;

	//2、改变预测偏移值的维度
	void *permuteLoc = nextWorkspacePtr((int8_t*)permuteConf, permuteConfSize);
	size_t permuteLoc1Size = floatSize(batchSize, featureSize * priorNum * 4);

	const int dimss = featureSize * priorNum * 4;
	const int GSS = (featureSize + BS - 1) / BS;
	permuteData2 << <GSS, BS >> > ((float*)locData, (float*)permuteLoc, dimss / priorNum / 4, 4, featureSize, priorNum, batchSize);



	//3、对预测置信度进行sigmoid
	void *sigmoidConf = nextWorkspacePtr((int8_t*)permuteLoc, permuteLoc1Size);
	size_t sigmoidSize = floatSize(batchSize, featureSize * priorNum * classNum);

	//4、indexPtr存放置信度对应的索引
	//sigmoidConf存放置信度
	void *indexPtr = nextWorkspacePtr((int8_t*)sigmoidConf, sigmoidSize);
	size_t indexSize = intSize(batchSize, featureSize * priorNum * classNum);
	sigmoid << <GS, BS >> > ((float*)permuteConf, (float *)sigmoidConf, (int*)indexPtr, featureSize * priorNum * classNum, batchSize);



	//5、对置信度进行排序
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;
	const int num_items = batchSize * featureSize * priorNum * classNum;
	const int num_segments = batchSize;

	void *d_offsets = nextWorkspacePtr((int8_t*)indexPtr, indexSize);
	size_t offsetSize = (num_segments + 1) * sizeof(int);

	setOffset << <1, 1 >> > ((int*)d_offsets, featureSize * priorNum * classNum, batchSize);


	//d_scoreSorted存放保存排序后的置信度
	void *d_scoreSorted = nextWorkspacePtr((int8_t*)d_offsets, offsetSize);
	size_t scoreSortedSize = floatSize(batchSize, featureSize * priorNum * classNum);
	//indexSorted存放排序后置信度在原始数组中的坐标
	void *indexSorted = nextWorkspacePtr((int8_t*)d_scoreSorted, scoreSortedSize);
	size_t indexSortedSize = intSize(batchSize, featureSize * priorNum * classNum);

	cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, (const float*)sigmoidConf, (float*)d_scoreSorted, (const int*)indexPtr, (int *)indexSorted,
		num_items, num_segments, (const int*)d_offsets, (const int*)d_offsets + 1, 0, sizeof(float) * 8,
		stream);
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, (const float*)sigmoidConf, (float*)d_scoreSorted, (const int*)indexPtr, (int *)indexSorted,
		num_items, num_segments, (const int*)d_offsets, (const int*)d_offsets + 1, 0, sizeof(float) * 8,
		stream);

	//test << <2, 1 >> > ((float*)confData, 1);
	//printf("-----------\n");

	//const int testdims = featureSize * priorNum * 2;
	//const int testBS = 128;
	//const int testGS = (testdims + testBS - 1) / testBS;
	//test << <testGS, testBS >> > ((float*)d_scoreSorted, testdims);

	//6、获取前keepK个得分、索引
	//predictProb保存前keepK个得分
	void *predictProb = nextWorkspacePtr((int8_t*)indexSorted, indexSortedSize);
	size_t predictProbSize = floatSize(batchSize, keepK);
	//predictIndex保存前keepK个得分所对应原始数组索引
	void *predictIndex = nextWorkspacePtr((int8_t*)predictProb, predictProbSize);
	size_t topkIndex = intSize(batchSize, keepK);
	//anchorIndex保存前keepK个得分所对应锚框的索引
	void *anchorIndex = nextWorkspacePtr((int8_t*)predictIndex, topkIndex);
	size_t anchorIndexSize = intSize(batchSize, keepK);
	//classIndex保存前keepK得分的预测框的预测类别
	void *classIndex = nextWorkspacePtr((int8_t*)anchorIndex, anchorIndexSize);
	size_t classIndexSize = intSize(batchSize, keepK);

	const int GS2 = (1000 + BS - 1) / BS;
	getTopkNum << <GS2, BS >> > ((const float*)d_scoreSorted, (const int*)indexSorted, (float*)predictProb, (int*)predictIndex, (float)scoreThreshold, keepK,
		(int*)anchorIndex, (int*)classIndex, classNum, batchSize, featureSize*priorNum*classNum);

	//7、获取预测偏移值以及对应锚框
	void *boxReg = nextWorkspacePtr((int8_t*)classIndex, classIndexSize);
	size_t boxRegSize = floatSize(batchSize, keepK * 4);

	void *anchors_i = nextWorkspacePtr((int8_t*)boxReg, boxRegSize);
	size_t anchorsSize = floatSize(batchSize, keepK * 4);



	//debug index
	const int testdimsindex = 80 * 80 * 9 * 2;
	const int testBsIndex = 128;
	const int testGsIndex = (testdimsindex + testBsIndex - 1) / testBsIndex;
	//test1 << <testGsIndex, testBsIndex >> > ((int*)indexSorted, testdimsindex);
	//anchorIndex中每一个batch的索引都是从0开始计数，batch对应的box索引需要加上前面batch的box数量
	getBoxAndAnchor << <GS2, BS >> > ((const int*)anchorIndex, (float*)anchor, (float*)permuteLoc, (float*)anchors_i, (float*)boxReg, keepK, batchSize, featureSize*priorNum * 4);

	void *afterDecode = nextWorkspacePtr((int8_t*)anchors_i, anchorsSize);
	size_t afterDecodeSize = floatSize(batchSize, keepK * 4);

	void* next = nextWorkspacePtr((int8_t*)afterDecode, afterDecodeSize);

	//8、解码，得到预测框
	decode << <GS2, BS >> > ((const float*)anchors_i, (const float*)boxReg, (float*)afterDecode, keepK, 4.1352, batchSize);

	//9、连接各层数组
	concatArray << <GS2, BS >> > ((const float*)afterDecode, (const float*)predictProb, (const int*)classIndex, (float*)boxPtr, (float*)scorePtr, (int*)classPtr, layerIndex, keepTopK, batchSize, keepK, layerNum, keepTopK);
	/*printf("%f\n", confData);*/
	cudaFree(d_temp_storage);
	return next;
}

__global__ void init_suppress(bool* suppress, bool* suppress_1d, int dims, int batchSize, int layerNum, int keepTopK)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	for (int i = 0; i < batchSize; i++)
	{
		suppress[i * dims + tid] = false;
		if (tid < (layerNum * keepTopK))
		{
			suppress_1d[i * layerNum * keepTopK + tid] = false;
		}
	}
}

__global__ void get_suppress(const bool* suppressIn, bool* suppressOut, int dims, int batchSize)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dims)
	{
		return;
	}
	for (int i = 0; i < batchSize; i++)
	{
		for (int j = 0; j < dims; j++)
		{
			if (suppressIn[i * dims * dims + j * dims + tid] == true)
			{
				//printf("111\n");
				suppressOut[i * dims + tid] = true;
				break;
			}
		}
	}
}

pluginStatus_t batchNms(cudaStream_t stream, int batchSize, void *workspace, const void *box, const void *score, const void *classIndex, float iouthreshold, int classNum, int layerNum, int topK,
	int srcW, int srcH, int tarW, int tarH, float* outLoc, float* outConf, int* outClass, int keepTopK)
{
	//1.排序，找出box最大值
	void     *d_temp_storage = NULL;
	void     *d_temp = NULL;
	size_t   temp_storage_bytes = 0;
	int num_segments = batchSize;

	void *sortedBox = workspace;
	size_t sortedBoxSize = predictDataSize(batchSize, layerNum * keepTopK * 4);

	void *offsets = nextWorkspacePtr((int8_t*)sortedBox, sortedBoxSize);
	size_t offsetSize = (num_segments + 1) * sizeof(int);
	//setoffset用于排序用
	setOffset << <1, 1 >> > ((int*)offsets, keepTopK * 4 * layerNum, batchSize);

	cub::DeviceSegmentedRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes, (const float*)box, (float*)sortedBox, batchSize * keepTopK * 4 * layerNum,
		num_segments, (const int*)offsets, (const int*)offsets + 1, 0, sizeof(float) * 8, stream);
	// Allocate temporary storage
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	// Run sorting operation
	cub::DeviceSegmentedRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes, (const float*)box, (float*)sortedBox, batchSize * keepTopK * 4 * layerNum,
		num_segments, (const int*)offsets, (const int*)offsets + 1, 0, sizeof(float) * 8, stream);

	void *boxForNmsPtr = nextWorkspacePtr((int8_t*)offsets, offsetSize);
	size_t boxForNmsSize = predictDataSize(batchSize, layerNum * keepTopK * 4);

	//求加上偏移值后的box
	//int sizePerBatch = layerNum * 1000;
	void *indexPtr = nextWorkspacePtr((int8_t*)boxForNmsPtr, boxForNmsSize);
	size_t indexPtrSize = indexDataSize(batchSize, layerNum * keepTopK);

	void *suppressBox = nextWorkspacePtr((int8_t*)indexPtr, indexPtrSize);
	size_t suppressBoxSize = boolDataSize(batchSize, layerNum * keepTopK * layerNum * keepTopK);

	void *suppress_1d = nextWorkspacePtr((int8_t*)suppressBox, suppressBoxSize);
	size_t suppress_1d_size = boolDataSize(batchSize, layerNum * keepTopK);

	//保存nms后前100个预测框的信息
	void *locAfterNms = nextWorkspacePtr((int8_t*)suppress_1d, suppress_1d_size);
	size_t locAfterNmsSize = predictDataSize(batchSize, layerNum*keepTopK * 4);

	const int BS = 128;
	const int GS = ((1000 * layerNum) + BS - 1) / BS;
	getBoxForNms << <GS, BS >> > ((float*)box, (int*)classIndex, (float*)sortedBox, (float*)boxForNmsPtr, batchSize, layerNum, layerNum * keepTopK, (int*)indexPtr, (int*)suppressBox, keepTopK);

	//nms
	//nms step1:分数排序，分数索引和box关联
	void* offsetScore = nextWorkspacePtr((int8_t*)locAfterNms, locAfterNmsSize);
	size_t offsetScoreSize = (num_segments + 1) * sizeof(int);
	setOffset << <1, 1 >> > ((int*)offsetScore, keepTopK * layerNum, batchSize);


	void *sortedScore = nextWorkspacePtr((int8_t*)offsetScore, offsetScoreSize);
	size_t sortedScoreSize = sigmoidDataSize(batchSize, layerNum * keepTopK);

	void *indexSorted = nextWorkspacePtr((int8_t*)sortedScore, sortedScoreSize);
	size_t indexSortedSize = indexDataSize(batchSize, layerNum * keepTopK);
	//std::cout << "sorted start" << std::endl;
	cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp, temp_storage_bytes, (const float*)score, (float*)sortedScore, (const int*)indexPtr, (int *)indexSorted,
		batchSize * keepTopK * layerNum, num_segments, (const int*)offsetScore, (const int*)offsetScore + 1, 0, sizeof(float) * 8,
		stream);

	// Allocate temporary storage
	cudaMalloc(&d_temp, temp_storage_bytes);

	//// Run sorting operation
	cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp, temp_storage_bytes, (const float*)score, (float*)sortedScore, (const int*)indexPtr, (int *)indexSorted,
		batchSize * keepTopK * layerNum, num_segments, (const int*)offsetScore, (const int*)offsetScore + 1, 0, sizeof(float) * 8,
		stream);

	int BS_INIT_SUPPRESS = 128;
	int GS_INIT_SUPPRESS = (layerNum * keepTopK * layerNum * keepTopK + BS_INIT_SUPPRESS - 1) / BS_INIT_SUPPRESS;
	init_suppress << <GS_INIT_SUPPRESS, BS_INIT_SUPPRESS >> > ((bool*)suppressBox, (bool*)suppress_1d, layerNum * keepTopK * layerNum * keepTopK, batchSize, layerNum, keepTopK);

	int GS_NMS = (layerNum * keepTopK + BS_INIT_SUPPRESS - 1) / BS_INIT_SUPPRESS;
	nms << <GS_NMS, BS_INIT_SUPPRESS >> > ((float*)boxForNmsPtr, (int*)indexSorted, (bool*)suppressBox, iouthreshold, layerNum * keepTopK, layerNum, keepTopK, batchSize);

	get_suppress << <GS_NMS, BS_INIT_SUPPRESS >> > ((bool*)suppressBox, (bool*)suppress_1d, (layerNum * keepTopK), batchSize);

	getResultAfterNms << <10, 800 >> > ((float*)box, (float*)score, (int*)classIndex, (int*)indexSorted, (bool*)suppress_1d,
		(float*)outConf, (float*)outLoc, (int*)outClass, topK, batchSize, layerNum, keepTopK);

	//float scaleW = float(srcW * 1.0 / tarW);
	//float scaleH = float(srcH * 1.0 / tarH);

	//scaleAndClip << <20, 512 >> > ((float*)locAfterNms, (float*)outLoc, srcW, srcH, scaleW, scaleH, topK, batchSize, layerNum, keepTopK);

	cudaFree(d_temp_storage);
	cudaFree(d_temp);
	return STATUS_SUCCESS;
} 