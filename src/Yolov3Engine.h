#include "Engine.h"

namespace engine
{
	class Yolov3Engine : public Engine
	{
	public:
		Yolov3Engine(const std::string &enginePath, const int gpuIndex);
		~Yolov3Engine();

		void preProcess(std::vector<imgData::_SHARE_IMAGE_DATA *> &images);
		void getResult(imgData::iDEFECT **result, const int batchIndex) const;

	private:
		int mSourceW;
		int mSourceH;
		int mTargetW;
		int mTargetH;

	};
}