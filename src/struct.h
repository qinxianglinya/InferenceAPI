#ifndef STRUCT_H
#define STRUCT_H
#include <vector>

namespace imgData
{



	typedef struct _iDEFECT {
		float nScore;//置信度
		int nType;//缺陷类型
		float rc[4];//缺陷坐标
		public: _iDEFECT()
		{
			nScore = 0;
			nType = -1;
			for (int i = 0; i < 4; i++)
			{
				rc[i] = 0;
			}
		}
	}iDEFECT;

	typedef struct _SHARE_IMAGE_DATA
	{
		int nWidth;		// 宽度
		int nHeight;	// 高度
		int nChannel;	// 通道数
		int nDepth;		// 深度
		int nLength;	// 图像的实际长度
		unsigned char *pData /*= nullptr*/;
		_SHARE_IMAGE_DATA()
		{
			nWidth = 0;
			nHeight = 0;
			nChannel = 0;
			nDepth = 0;
			nLength = 0;
		}
		~_SHARE_IMAGE_DATA()
		{
			pData = NULL;
		}
	}SHARE_IMAGE_DATA;

	typedef struct _Coordinate
	{
		long left;
		long top;
		long right;
		long bottom;
	}Coordinate;
}

namespace myPlugin
{
	struct GridAnchorParametersTf
	{
		int level, scalesPerOctave;
		// float anchorScale;
		float* aspectRatios;
		int numAspectRatios;
		int imgH, imgW;
		int W, H;
		float variance[4];
	};

	struct Coordinate
	{
		float x0;
		float y0;
		float x1;
		float y1;
	};

	struct FeatureSize
	{
		int nWidth;
		int nHeight;
	};

	struct AnchorParamsTorch
	{
		int nOffset;
		int nStride;
		float nSize[3];
		float fAspectRatio[3] /*= { 0.5, 1, 2 }*/;
		Coordinate fBaseAnchor[9];
		FeatureSize featureSize;
		AnchorParamsTorch()
		{
			fAspectRatio[0] = 0.5;
			fAspectRatio[1] = 1;
			fAspectRatio[2] = 2;
		}
	};

	struct DetectionOutputParametersTorch
	{
		int nbCls, topK, keepTopK;
		int nbLayer;
		int nbPriorbox;
		int srcW, srcH, targetW, targetH;
		float scoreThreshold, iouThreshold;
	};

	struct Yolov3NmsParams
	{
		int nbCls;
		float conf_thr, score_thr, iou_thr;
		float factor_scales[4];
		FeatureSize featureSize[3];
		int stride[3];
	};

	struct EngineParams
	{
		int imgW;
		int imgH;
		int imgC;

		std::vector<std::string> inputTensorNames;
		std::vector<std::string> outputTensorNames;
		std::vector<std::vector<int>> inputDims;

		bool FP32;
		bool FP16;
		int maxBatchSize;
		std::string trtSavePath;
		EngineParams()
		{
			imgW = 640;
			imgH = 640;
			imgC = 3;

			FP32 = false;
			FP16 = false;
			maxBatchSize = 16;
		}
	};

	struct Yolov3Params : public EngineParams
	{
		std::string weightsPath;
		Yolov3Params()
		{
			weightsPath = "";
		}
	};

	enum NetworkPart
	{
		BACKBONE = 1,
		NECK,
		HEAD
	};

	struct Coordinate2d
	{
		float x, y;
	};

	struct AnchorParamsYolov3
	{
		Coordinate fBaseAnchor[3];
		Coordinate2d fBaseSize[3];
		FeatureSize featureSize;
		int nStride;
	};
}
//struct TParams
//{
//	int nSourceW = 0;
//	int nSourceH = 0;
//	int nTargetW = 0;
//	int nTargetH = 0;
//	int nChannel = 3;
//	int nKeepTop = 100;
//	float fVisualThreshold = 0.3;
//};


#endif // !STRUCT_H

