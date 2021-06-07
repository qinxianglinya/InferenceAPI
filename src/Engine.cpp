#include "Engine.h"
#include <iostream>
#include <fstream>

namespace engine
{
	Engine::Engine(const std::string & enginePath, const int gpuIndex) :
		mEnginePath(enginePath), mGpuIndex(gpuIndex)
	{
		initLibNvInferPlugins(&mLogger, "");
	}

	Engine::~Engine()
	{
		
		//if (mStream) cudaStreamDestroy(mStream);
		//if (mRuntime) mRuntime->destroy();
		//if (mContext) mContext->destroy();
	}

	void Engine::initContext()
	{
		load(mEnginePath);
		prepare();
	}

	void Engine::setGpuIndex(const int & index)
	{
		mGpuIndex = index;
	}

	void Engine::load(const std::string & path)
	{
		cudaSetDevice(mGpuIndex);
		std::ifstream file(path, std::ios::in | std::ios::binary);
		file.seekg(0, file.end);
		size_t size = file.tellg();
		file.seekg(0, file.beg);

		char *buffer = new char[size];
		file.read(buffer, size);
		file.close();

		mRuntime = createInferRuntime(mLogger);
		mEngine = std::shared_ptr<ICudaEngine>(mRuntime->deserializeCudaEngine(buffer, size), InferDeleter());
		if (mRuntime) mRuntime->destroy();
		delete[] buffer;
	}

	void Engine::prepare()
	{
		cudaSetDevice(mGpuIndex);
		mContext = std::shared_ptr<IExecutionContext>(mEngine->createExecutionContext(), InferDeleter());
	}

	//同步推断
	bool Engine::synchInfer()
	{
		//std::cout << "synch infer" << std::endl;
		cudaSetDevice(mGpuIndex);

		if (!mContext->execute(mBuffer->getBatchSize(), mBuffer->getDeviceBindings().data()))
		{
			return false;
		}
		mBuffer->copyOutputToHost();

		return true;
	}

	bool Engine::startAsyncInfer()
	{
		//std::cout << getBatchSize() << std::endl;
		//cudaSetDevice(mGpuIndex);
		//cudaStreamCreate(&mStream);
		//for (int i = 0; i < mBuffers.size(); i++)
		//{
		//	if (!mContext->enqueue(mBuffers[i]->getBatchSize(), mBuffers[i]->getDeviceBindings().data(), mStream, nullptr))
		//	{
		//		return false;
		//	}
		//	mBuffers[i]->copyOutputToHostAsync(mStream);
		//}
		return true;
	}

	bool Engine::finishAsyncInfer()
	{
		//cudaSetDevice(mGpuIndex);
		//cudaStreamSynchronize(mStream);
		return true;
	}

	int Engine::getBatchSize() const
	{
		return mEngine->getMaxBatchSize();
		//return 2;
	}

	void Engine::assignImgByBatch(std::vector<imgData::_SHARE_IMAGE_DATA*>& imgPerGpu, std::vector<std::vector<imgData::_SHARE_IMAGE_DATA*>>& imgByBatch)
	{
		int imgNum = imgPerGpu.size();
		const int maxBatchTimes = imgNum / getBatchSize();
		//const int maxBatchTimes = imgNum / 8;
		if (maxBatchTimes == 0)//如果待处理图片总数不足一个batch，则一次性处理所有图片
		{
			std::vector<imgData::_SHARE_IMAGE_DATA*> images;
			for (int i = 0; i < imgNum; i++)
			{
				images.push_back(imgPerGpu[i]);
			}
			imgByBatch.push_back(images);
		}
		else
		{
			const int rest = imgNum - maxBatchTimes * getBatchSize();
			for (int i = 0; i < maxBatchTimes; i++)
			{
				std::vector<imgData::_SHARE_IMAGE_DATA*> images;
				for (int j = 0; j < getBatchSize(); j++)
				{
					images.push_back(imgPerGpu[i * getBatchSize() + j]);
				}
				imgByBatch.push_back(images);
			}
			if (rest != 0)
			{
				std::vector<imgData::_SHARE_IMAGE_DATA*> images;
				for (int i = maxBatchTimes * getBatchSize(); i < maxBatchTimes * getBatchSize() + rest; i++)
				{
					images.push_back(imgPerGpu[i]);
				}
				imgByBatch.push_back(images);
			}
		}
	}

}



