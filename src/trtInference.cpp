#include "trtInference.h"
#include "retinanetTf.h"
#include "RetinaNetDetectron.h"
#include "Engine.h"
#include "Yolov3Engine.h"


using namespace retinanetTf;

std::mutex mtx;
static std::vector<Engine*> m_Engines = {};
static int m_usedGpuNum = 0;
static int m_threadNum = 0;
static uint8_t** d_BigImg = nullptr;
static void assignImgForGpu(std::vector<imgData::_SHARE_IMAGE_DATA*> &totalImg, std::vector<std::vector<imgData::_SHARE_IMAGE_DATA*>> &imgGpus);
static void synchInfer(Engine* engine, std::vector<imgData::_SHARE_IMAGE_DATA*>image, imgData::iDEFECT **result);
static void asyncInfer(std::shared_ptr<engine::Engine> engine, std::vector<imgData::_SHARE_IMAGE_DATA*>image, imgData::iDEFECT **result);

bool TrtInference::initEngines(const char* trtPath)
{
	TrtInference::releaseEngine();

	int gpuNums = 0;
	cudaGetDeviceCount(&gpuNums);

	m_usedGpuNum = gpuNums;

	std::string modelPath(trtPath);
	int pos = modelPath.find_last_of('/');
	std::string modelName(modelPath.substr(pos + 1));
	for (int i = 0; i < gpuNums; i++)
	{
		if (modelName == "tf-retinanet.trt")
		{
			Engine* engine = new retinanetTf::RetinaNetTf(trtPath, i);
			engine->initContext();
			m_Engines.push_back(engine);
		}
		else if (modelName == "d2-retinanet.trt")
		{
			//std::shared_ptr<Engine> engine(new RetinaNetDetectron(trtPath, i));
			Engine* engine = new RetinaNetDetectron(trtPath, i);
			engine->initContext();
			m_Engines.push_back(engine);
		}
		else if (modelName == "d2-yolov3.trt")
		{
			Engine* engine = new Yolov3Engine(trtPath, i);
			engine->initContext();
			m_Engines.push_back(engine);
		}
		else
		{
			std::cout << "model " << modelName << "is not supported now!" << std::endl;
			return false;
		}
	}

	return true;
}

//bool TrtInference::uploadImageToGpu(imgData::_SHARE_IMAGE_DATA * input)
//{
//	int imgSize = input->nWidth * input->nHeight * input->nDepth;
//	int size[10] = {0};
//	int average = imgSize / m_usedGpuNum;
//	int rest = imgSize - (m_usedGpuNum - 1) * average;
//	for (int i = 0; i < (m_usedGpuNum - 1); i++)
//	{
//
//	}
//	for(int i=0; i<m_usedGpuNum; i++)
//	{
//		cudaSetDevice(i);
//		cudaMalloc((void**)&d_BigImg[i], input->nWidth * input->nHeight * input->nDepth * sizeof(uint8_t));
//		cudaMemcpy((void*)d_BigImg[i], (void*)(*input).pData, input->nWidth * input->nHeight * input->nDepth * sizeof(uint8_t), cudaMemcpyHostToDevice);
//	}
//
//	return true;
//}

bool TrtInference::doInference(imgData::_SHARE_IMAGE_DATA *input, const int &totalNum, imgData::iDEFECT** result, bool isSynchronized)
{
	std::vector<imgData::_SHARE_IMAGE_DATA*> imgTotal;
	std::vector<std::vector<imgData::_SHARE_IMAGE_DATA*>> imgGpus;

	for (int i = 0; i < totalNum; i++)
	{
		imgTotal.push_back(&input[i]);
	}
	assignImgForGpu(imgTotal, imgGpus);

	imgData::iDEFECT ***tempResult;
	tempResult = new imgData::iDEFECT**[m_usedGpuNum];
	for (int i = 0; i < m_usedGpuNum; i++)
	{
		tempResult[i] = new imgData::iDEFECT *[imgGpus[i].size()];
		for (int j = 0; j < imgGpus[i].size(); j++)
		{
			tempResult[i][j] = new imgData::iDEFECT[1024];
		}
	}


	for (int i = 0; i < imgGpus.size(); i++)
	{
		if (isSynchronized)
		{
			std::thread thread(&synchInfer, m_Engines[i], imgGpus[i], tempResult[i]);
			thread.detach();
		}
		else
		{
		}
	}

	while (m_threadNum != m_usedGpuNum)
	{
		std::this_thread::sleep_for(std::chrono::nanoseconds(1));
	}

	for (int i = 0; i < m_usedGpuNum - 1; i++)
	{
		for (int j = 0; j < imgGpus[i].size(); j++)
		{
			for (int k = 0; k < 1024; k++)
			{
				result[i*imgGpus[i].size() + j][k] = tempResult[i][j][k];
			}

		}
	}
	for (int i = 0; i < imgGpus[m_usedGpuNum - 1].size(); i++)
	{
		for (int j = 0; j < 1024; j++)
		{
			result[(m_usedGpuNum - 1) * imgGpus[0].size() + i][j] = tempResult[m_usedGpuNum - 1][i][j];
		}
	}

	for (int i = 0; i < m_usedGpuNum; i++)
	{
		for (int j = 0; j < imgGpus[i].size(); j++)
		{
			delete[]tempResult[i][j];
		}
		delete[] tempResult[i];
	}
	delete[]tempResult;
	m_threadNum = 0;

	//ÊÍ·ÅÏÔ´æ
	return true;
}

//bool TrtInference::doInference(imgData::Coordinate * rects, const int totalNum, imgData::iDEFECT ** result, int & resultNum, imgData::_SHARE_IMAGE_DATA * defectImg, int * imgIndex, bool isSynchronized)
//{
//	//m_usedGpuNum
//
//
//	cudaFree(d_BigImg);
//	return true;
//}

void TrtInference::releaseEngine()
{
	for (int i = 0; i < m_Engines.size(); i++)
	{
		if (m_Engines[i] != nullptr)
		{
			delete m_Engines[i];
			m_Engines[i] = nullptr;
		}
	}
	m_Engines.resize(0);
}


void assignImgForGpu(std::vector<imgData::_SHARE_IMAGE_DATA*>& totalImg, std::vector<std::vector<imgData::_SHARE_IMAGE_DATA*>>& imgGpus)
{
	int gpuNum = m_usedGpuNum;
	int imgNum = totalImg.size();
	int imgNumPerGpu = imgNum / gpuNum;
	int rest = imgNum - imgNumPerGpu * gpuNum;
	if (gpuNum == 1)
	{
		imgGpus.push_back(totalImg);
	}
	else
	{
		for (int i = 0; i < (gpuNum - 1); i++)
		{
			std::vector<imgData::_SHARE_IMAGE_DATA*> image;
			for (int j = 0; j < imgNumPerGpu; j++)
			{
				image.push_back(totalImg[i*imgNumPerGpu + j]);
			}
			imgGpus.push_back(image);
		}
		int start = imgNumPerGpu * (gpuNum-1);
		std::vector<imgData::_SHARE_IMAGE_DATA*> image;
		while (start != imgNum)
		{
			image.push_back(totalImg[start]);
			start++;
		}
		imgGpus.push_back(image);
	}
}


void synchInfer(engine::Engine* engine, std::vector<imgData::_SHARE_IMAGE_DATA*>image, imgData::iDEFECT **result)
{
	std::vector<std::vector<imgData::_SHARE_IMAGE_DATA*>> imgByBatch;
	engine->assignImgByBatch(image, imgByBatch);
	
	for (int i = 0; i < imgByBatch.size(); i++)
	{
		engine->preProcess(imgByBatch[i]);
		engine->synchInfer();
		engine->getResult(result, i);
	}

	mtx.lock();
	m_threadNum++;
	mtx.unlock();
}

void asyncInfer(std::shared_ptr<engine::Engine> engine, std::vector<imgData::_SHARE_IMAGE_DATA*>image, imgData::iDEFECT **result)
{
}

