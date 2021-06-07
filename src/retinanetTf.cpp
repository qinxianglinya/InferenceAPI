#include "retinanetTf.h"
#include <Windows.h>
namespace retinanetTf
{
	RetinaNetTf::RetinaNetTf(const std::string &enginePath, const int gpuIndex) :Engine(enginePath, gpuIndex)
	{
	}

	RetinaNetTf::~RetinaNetTf()
	{
	}

	/*
	std::vector<std::vector<imgData::_SHARE_IMAGE_DATA>> images：将每个GPU所要处理的图片总数按照batchSize进行划分，分组之后输入images。
	预处理部分包含：将图片像素值存放进一维数组，并进行resize操作，这两个部分目前均是在GPU上进行。
	*/
	void RetinaNetTf::preProcess(std::vector<imgData::_SHARE_IMAGE_DATA*> &images)
	{
		cudaSetDevice(mGpuIndex);
		//cudaStreamCreate(&mStream);

		mSourceW = images[0]->nWidth;
		mSourceH = images[0]->nHeight;

		Dims dim = mEngine->getBindingDimensions(0);
		mTargetH = dim.d[1];
		mTargetW = dim.d[2];

		//std::vector<std::vector<imgData::_SHARE_IMAGE_DATA*>> imgByBatch;

		//assignImgByBatch(images, imgByBatch);


		//mBuffer = new BufferManager(mEngine, images.size());
		mBuffer = std::make_shared<BufferManager>(mEngine, images.size());

		float *dInputBuffer = static_cast<float *>(mBuffer->getDeviceBuffer("Input"));
		uint8_t *inputGpu;
		float *resizedOutputGpu;
		cudaMalloc((void**)&inputGpu, images.size() * mSourceW * mSourceH * 3 * sizeof(uint8_t));
		cudaMalloc((void**)&resizedOutputGpu, images.size() * mTargetW * mTargetH * 3 * sizeof(float));
		for (int j = 0; j < images.size(); j++)
		{
			uint8_t* temp;
			cudaMalloc((void**)&temp, mSourceW * mSourceH * 3 * sizeof(uint8_t));
			cudaMemcpy(temp, (*images[j]).pData, mSourceW * mSourceH * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice);
			copyImg(temp, inputGpu, j, mSourceW * mSourceH);
			cudaFree(temp);
		}
		float mean1 = 123.68;
		float mean2 = 116.779;
		float mean3 = 103.939;
		resizeAndNorm(inputGpu, resizedOutputGpu, dInputBuffer, images.size() * mTargetW * mTargetH, mTargetW,
			mTargetH, mSourceW, mSourceH, mean1, mean2, mean3);
		cudaFree(inputGpu);
		cudaFree(resizedOutputGpu);


	}

	void RetinaNetTf::getResult(imgData::iDEFECT ** result, const int batchIndex) const
	{
		cudaSetDevice(mGpuIndex);


		const float* detectionOut = static_cast<const float*>(mBuffer->getHostBuffer("NMS"));
		const int* keepCount = static_cast<const int*>(mBuffer->getHostBuffer("NMS_1"));

		//std::cout << "post" << std::endl;
		for (int j = 0; j < mBuffer->getBatchSize(); j++)
		{
			int numDetections = 0;
			//std::cout << "keepCount:" << "j:"<<j<<"..."<<keepCount[i] << std::endl;
			for (int k = 0; k < keepCount[j]; k++)
			{
				const float *det = &detectionOut[0] + (j * 100 + k) * 7;
				if (det[2] < 0.3)
				{
					continue;
				}
				int detection = det[1];
				//assert(detection < outputClsSize);

				int imgIndex = batchIndex * mBuffer->getBatchSize() + j;
				result[imgIndex][numDetections].nScore = det[2];
				result[imgIndex][numDetections].nType = detection;
				result[imgIndex][numDetections].rc[0] = det[3] * mSourceW;
				result[imgIndex][numDetections].rc[1] = det[4] * mSourceH;
				result[imgIndex][numDetections].rc[2] = det[5] * mSourceW;
				result[imgIndex][numDetections].rc[3] = det[6] * mSourceH;
				numDetections++;
			}
		}
		//if (mBuffer != nullptr)
		//{
		//	delete mBuffer;
		//	//mBuffer = nullptr;
		//}
	}


}


