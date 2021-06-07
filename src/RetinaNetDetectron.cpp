#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "RetinaNetDetectron.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

std::mutex mtx1;
float clamp_(float data, int limitMax)
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


RetinaNetDetectron::RetinaNetDetectron(const std::string & enginePath, const int gpuIndex) :Engine(enginePath, gpuIndex)
{


}

RetinaNetDetectron::~RetinaNetDetectron()
{
}


void RetinaNetDetectron::preProcess(std::vector<imgData::_SHARE_IMAGE_DATA*>& images)
{
	cudaSetDevice(mGpuIndex);
	//cudaStreamCreate(&mStream);

	mSourceW = images[0]->nWidth;
	mSourceH = images[0]->nHeight;

	Dims dim = mEngine->getBindingDimensions(0);
	mTargetH = dim.d[1];
	mTargetW = dim.d[2];

	int alignByte = 4;
	int dw = alignByte - ((mSourceW * 3) % alignByte);
	int alignSize = (mSourceW * 3 + dw) * mSourceH;


	mBuffer = std::make_shared<BufferManager>(mEngine, images.size());
	float *dInputBuffer = static_cast<float *>(mBuffer->getDeviceBuffer("data"));
	uint8_t *inputGpu;
	float *resizedOutputGpu;
	float * transformGpu;

	cudaMalloc((void**)&inputGpu, images.size() * mSourceW * mSourceH * 3 * sizeof(uint8_t));
	cudaMalloc((void**)&resizedOutputGpu, images.size() * mTargetW * mTargetH * 3 * sizeof(float));
	cudaMalloc((void**)&transformGpu, images.size() * mTargetW * mTargetH * 3 * sizeof(float));
	for (int j = 0; j < images.size(); j++)
	{
		bool align = true;
		if (align)
		{
			uint8_t* temp;
			cudaMalloc((void**)&temp, alignSize * sizeof(uint8_t));
			cudaMemcpy((void*)temp, (void*)((char*)(*images[j]).pData), alignSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
			copyImgAlign(temp, inputGpu, j, mSourceW, mSourceH, dw);
			cudaFree(temp);
		}
		else
		{
			uint8_t* temp;
			cudaMalloc((void**)&temp, mSourceW * mSourceH * 3 * sizeof(uint8_t));
			cudaMemcpy((void*)temp, (void*)((char*)(*images[j]).pData), mSourceW * mSourceH * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice);
			copyImgD2(temp, inputGpu, j, mSourceW * mSourceH);
			cudaFree(temp);
		}
	}
	float mean1 = 123.675;
	float mean2 = 116.280;
	float mean3 = 103.530;

	float std1 = 1;
	float std2 = 1;
	float std3 = 1;


	resizeAndNormD2(inputGpu, resizedOutputGpu, mTargetW, mTargetH, mSourceW, mSourceH, images.size());
	transformD2(resizedOutputGpu, transformGpu, dInputBuffer, images.size(), mTargetW * mTargetH, mean1, mean2, mean3, std1, std2, std3);

	cudaFree(inputGpu);
	cudaFree(resizedOutputGpu);
	cudaFree(transformGpu);
}



void RetinaNetDetectron::getResult(imgData::iDEFECT** result, const int batchIndex) const
{
	cudaSetDevice(mGpuIndex);

	const float* location = static_cast<const float*>(mBuffer->getHostBuffer("Loc"));
	const float* score = static_cast<const float*>(mBuffer->getHostBuffer("Score"));
	const int* clsIndex = static_cast<const int*>(mBuffer->getHostBuffer("Cls"));
	int topK = mEngine->getBindingDimensions(2).d[1];
	float scaleW = float(mSourceW * 1.0 / mTargetW);
	float scaleH = float(mSourceH * 1.0 / mTargetH);

	for (int j = 0; j < mBuffer->getBatchSize(); j++)
	{
		int numDetections = 0;
		for (int k = 0; k < topK; k++)
		{
			if (score[j*topK + k] != (0.0f))
			{
				int imgIndex = batchIndex * getBatchSize() + j;
				result[imgIndex][numDetections].nScore = score[j*topK + k];
				result[imgIndex][numDetections].nType = clsIndex[j*topK + k];
				float xmin = location[j*topK * 4 + k * 4];
				float ymin = location[j*topK * 4 + k * 4 + 1];
				float xmax = location[j*topK * 4 + k * 4 + 2];
				float ymax = location[j*topK * 4 + k * 4 + 3];
				float _xmin = 0, _ymin = 0, _xmax = 0, _ymax = 0;
				_xmin = xmin * scaleW;
				_ymin = ymin * scaleH;
				_xmax = xmax * scaleW;
				_ymax = ymax * scaleH;
				_xmin = clamp_(_xmin, mSourceW);
				_ymin = clamp_(_ymin, mSourceH);
				_xmax = clamp_(_xmax, mSourceW);
				_ymax = clamp_(_ymax, mSourceH);
				result[imgIndex][numDetections].rc[0] = _xmin;
				result[imgIndex][numDetections].rc[1] = _ymin;
				result[imgIndex][numDetections].rc[2] = _xmax;
				result[imgIndex][numDetections].rc[3] = _ymax;
				numDetections++;
			}
		}
	}
	//size_t free;
	//size_t total;
	//CUresult uRet = cuMemGetInfo(&free, &total);
	//mtx1.lock();
	//if (uRet == CUDA_SUCCESS)
	//	printf("free = %dM\ntotal = %dM", free / 1024 / 1024, total / 1024 / 1024);
	//mtx1.unlock();
}