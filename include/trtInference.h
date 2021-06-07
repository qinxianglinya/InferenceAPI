#ifndef TRTINFERENCE_H
#define TRTINFERENCE_H

//#include "struct.h"
//#include "retinanetTf.h"
//#include "RetinaNetDetectron.h"
#include "struct.h"

//Singleton

class _declspec(dllexport) TrtInference
{
public:
	//modelName默认为retinanet,如果存在
	static bool initEngines(const char* modelName);
	//static bool uploadImageToGpu(imgData::_SHARE_IMAGE_DATA* input);
	static bool doInference(imgData::_SHARE_IMAGE_DATA* input, const int &totalNum, imgData::iDEFECT **result, bool isSynchronized = true);
	//static bool doInference(imgData::Coordinate* rects, const int totalNum, imgData::iDEFECT** result, int& resultNum, imgData::_SHARE_IMAGE_DATA* defectImg,
	//	int* imgIndex, bool isSynchronized = true);
	static void releaseEngine();
};

#endif // !TRTINFERENCE_H
