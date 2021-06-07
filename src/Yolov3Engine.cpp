#include "Yolov3Engine.h"

engine::Yolov3Engine::Yolov3Engine(const std::string & enginePath, const int gpuIndex) : Engine(enginePath, gpuIndex)
{
}

engine::Yolov3Engine::~Yolov3Engine()
{
}

void engine::Yolov3Engine::preProcess(std::vector<imgData::_SHARE_IMAGE_DATA*>& images)
{
	cudaSetDevice(mGpuIndex);
	cudaStreamCreate(&mStream);

	mSourceW = images[0]->nWidth;
	mSourceH = images[0]->nHeight;


	Dims dim = mEngine->getBindingDimensions(0);
	int mTargetH = dim.d[1];
	int mTargetW = dim.d[2];

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
		uint8_t* temp;
		cudaMalloc((void**)&temp, mSourceW * mSourceH * 3 * sizeof(uint8_t));
		cudaMemcpy((void*)temp, (void*)(*images[j]).pData, mSourceW * mSourceH * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice);
		copyImgD2(temp, inputGpu, j, mSourceW * mSourceH);
		cudaFree(temp);
	}

	float mean1 = 0;
	float mean2 = 0;
	float mean3 = 0;

	float std1 = 255;
	float std2 = 255;
	float std3 = 255;

	resizeAndNormD2(inputGpu, resizedOutputGpu, mTargetW, mTargetH, mSourceW, mSourceH, images.size());
	transformYolov3(resizedOutputGpu, transformGpu, dInputBuffer, images.size(), mTargetW * mTargetH, mean1, mean2, mean3, std1, std2, std3);


	//uint8_t* h_input1 = new uint8_t[1354752];
	//float* h_input = new float[1354752];
	//cudaMemcpy(h_input, dInputBuffer, 1354752 * sizeof(float), cudaMemcpyDeviceToHost);
	//for (int i = 0; i < 20; i++)
	//{
	//	std::cout << "input:" << h_input[i] << std::endl;
	//}


	cudaFree(inputGpu);
	cudaFree(resizedOutputGpu);
	cudaFree(transformGpu);
}

void engine::Yolov3Engine::getResult(imgData::iDEFECT ** result, const int batchIndex) const
{
	cudaSetDevice(mGpuIndex);

	const float* location = static_cast<const float*>(mBuffer->getHostBuffer("box"));
	const int* cls = static_cast<const int*>(mBuffer->getHostBuffer("cls"));
	const float* score = static_cast<const float*>(mBuffer->getHostBuffer("conf"));

	for (int j = 0; j < mBuffer->getBatchSize(); j++)
	{
		int numDetections = 0;
		for (int k = 0; k < 100; k++)
		{
			if (score[j * 100 + k] != (0.0f))
			{
				//std::cout << score[j * 100 + k] << std::endl;
				int imgIndex = batchIndex * getBatchSize() + j;
				result[imgIndex][numDetections].nScore = score[j * 100 + k];
				result[imgIndex][numDetections].nType = cls[j * 100 + k];
				result[imgIndex][numDetections].rc[0] = location[j * 100 * 4 + k * 4];
				result[imgIndex][numDetections].rc[1] = location[j * 100 * 4 + k * 4 + 1];
				result[imgIndex][numDetections].rc[2] = location[j * 100 * 4 + k * 4 + 2];
				result[imgIndex][numDetections].rc[3] = location[j * 100 * 4 + k * 4 + 3];
				numDetections++;
			}
		}
	}
}
