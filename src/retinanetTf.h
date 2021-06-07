#ifndef RETINANET_H
#define RETINANET_H
#include "Engine.h"
using namespace engine;

namespace retinanetTf
{

	class RetinaNetTf : public Engine
	{
	public:
		RetinaNetTf(const std::string &enginePath, const int gpuIndex);
		~RetinaNetTf();

	public:
		void preProcess(std::vector<imgData::_SHARE_IMAGE_DATA *> &images);
		void getResult(imgData::iDEFECT **result, const int batchIndex) const;
	private:
		int mSourceW;
		int mSourceH;
		int mTargetW;
		int mTargetH;
	};
}

#endif // !RETINANET_H
