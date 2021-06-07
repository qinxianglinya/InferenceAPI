#ifndef ENGINE
#define ENGINE

#include <NvInfer.h>
#include <string>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <memory>
#include "buffer.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <mutex>
#include "struct.h"
#include "NvInferPlugin.h"


using namespace nvinfer1;
using namespace buffer;

extern "C" void resizeAndNorm(void* inputGpu, void* outputGpu, void* normGpu, int size, int dstW, int dstH, int srcW, int srcH, float mean1, float mean2, float mean3);

extern "C" void copyImgD2(uint8_t* input, uint8_t* output, int index, int k);
extern "C" void resizeAndNormD2(uint8_t* inputGpu, float* resizedOutputGpu,
	int dstW, int dstH, int srcW, int srcH, int batchSize);

extern "C" void transformD2(void* resizedInput, void* transform, void* norm, int batchSize, int dims, float mean1, float mean2, float mean3, float std1, float std2, float std3);
extern "C" void transformYolov3(void* resizedInput, void* transform, void* norm, int batchSize, int dims, float mean1, float mean2, float mean3, float std1, float std2, float std3);
extern "C" void resizeAndNorm_torch(void* inputGpu, void* outputGpu, void* normGpu, int size, int dstW, int dstH, int srcW, int srcH, float mean1, float mean2, float mean3);
extern "C" void copyImg(void* input, void* output, int index, int dim);
extern "C" void padding(void* input, void* output, int resizedW, int resizedH, int batchSize);
extern "C" void copyImgAlign(uint8_t* input, uint8_t* output, int index, int srcW, int srcH, int dw);

namespace engine
{
	class CLogger : public nvinfer1::ILogger {
		void log(Severity severity, const char* msg) override
		{
			// suppress info-level messages
			//std::cout << msg << std::endl;
		/*	if (severity != Severity::kVERBOSE)
				std::cout << msg << std::endl;*/
		}
	};


	class Engine
	{
	public:
		Engine(const std::string &enginePath, const int gpuIndex);
		virtual ~Engine();
	public:
		void initContext();
		//同步推断
		bool synchInfer();

		//异步推断
		bool startAsyncInfer();
		bool finishAsyncInfer();

		virtual void preProcess(std::vector<imgData::_SHARE_IMAGE_DATA*> &images) = 0;

		virtual void getResult(imgData::iDEFECT** result, const int batchIndex)const = 0;

	public:
		int getBatchSize() const;
		void setGpuIndex(const int &index);

	protected:
		cudaStream_t mStream;
		std::shared_ptr<ICudaEngine> mEngine = nullptr;
		std::shared_ptr<BufferManager> mBuffer = nullptr;
		int mGpuIndex;

	private:
		CLogger mLogger;
		IRuntime *mRuntime = nullptr;
		std::shared_ptr<IExecutionContext> mContext = nullptr;
		//IExecutionContext* mContext = nullptr;
		std::string mEnginePath;

		void load(const std::string &path);
		void prepare();
	public:
		void assignImgByBatch(std::vector<imgData::_SHARE_IMAGE_DATA*> &imgPerGpu, std::vector<std::vector<imgData::_SHARE_IMAGE_DATA*>>&imgByBatch);
		
	};

	struct InferDeleter
	{
		template <typename T>
		void operator()(T* obj) const
		{
			if (obj)
			{
				obj->destroy();
			}
		}
	};

}

#endif // !ENGINE

