#pragma once
#include "Engine.h"
using namespace engine;

class RetinaNetDetectron :public Engine
{
public:
	RetinaNetDetectron(const std::string &enginePath, const int gpuIndex);
	~RetinaNetDetectron();

public:
	void preProcess(std::vector<imgData::_SHARE_IMAGE_DATA *> &images);
	//void getResult(std::vector<imgData::iDEFECT*>&result, const int batchIndex)const;
	void getResult(imgData::iDEFECT **result, const int batchIndex) const;

//private:
//	std::vector<std::shared_ptr<BufferManager>> mBuffer;
//	std::vector<BufferManager> mBuf;

private:
	int mSourceW;
	int mSourceH;
	int mTargetW;
	int mTargetH;
};

