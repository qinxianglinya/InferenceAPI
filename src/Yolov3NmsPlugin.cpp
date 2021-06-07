#include "Yolov3NmsPlugin.h"
#include "bboxUtils.h"
#include <ctime>
#include "nmsUtils.h"
namespace
{
	const char* NMS_YOLOV3_PLUGIN_VERSION{ "1" };
	const char* NMS_YOLOV3_PLUGIN_NAME{ "NMS_YOLOV3_TRT" };
} // namespace

detectron2::Yolov3NmsPlugin::Yolov3NmsPlugin(Yolov3NmsParams params, const int* featureSizeIn)
{
	mParams.nbCls = params.nbCls;
	mParams.conf_thr = params.conf_thr;
	mParams.score_thr = params.score_thr;
	mParams.iou_thr = params.iou_thr;
	for (int i = 0; i < 4; i++)
	{
		mParams.factor_scales[i] = params.factor_scales[i];
	}
	mFeatureSize.resize(3);
	for (int i = 0; i < 3; i++)
	{
		mFeatureSize[i] = featureSizeIn[i];
		mParams.stride[i] = params.stride[i];
	}
}

detectron2::Yolov3NmsPlugin::Yolov3NmsPlugin(const void * data, size_t length)
{
	const char *d = reinterpret_cast<const char*>(data), *a = d;
	mParams.nbCls = read<int>(d);
	mParams.conf_thr = read<float>(d);
	mParams.score_thr = read<float>(d);
	mParams.iou_thr = read<float>(d);
	for (int i = 0; i < 4; i++)
	{
		mParams.factor_scales[i] = read<float>(d);
	}
	mFeatureSize.resize(3);
	for (int i = 0; i < 3; i++)
	{
		mFeatureSize[i] = read<int>(d);
	}
	for (int i = 0; i < 3; i++)
	{
		mParams.stride[i] = read<int>(d);
	}
}

int detectron2::Yolov3NmsPlugin::getNbOutputs() const
{
	return 3;
}

Dims detectron2::Yolov3NmsPlugin::getOutputDimensions(int index, const Dims * inputs, int nbInputDims)
{
	if (index == 0)
	{
		return DimsCHW(1, 100 * 4, 1);
	}
	else if (index == 1)
	{
		return DimsCHW(1, 100, 1);
	}
	else
	{
		return DimsCHW(1, 100, 1);
	}
}

int detectron2::Yolov3NmsPlugin::initialize()
{
	return 0;
}

void detectron2::Yolov3NmsPlugin::terminate()
{
}

void detectron2::Yolov3NmsPlugin::destroy()
{
	delete this;
}

size_t detectron2::Yolov3NmsPlugin::getWorkspaceSize(int batchSize) const
{
	return yolov3NmsWorkspaceSize(batchSize, mParams, mFeatureSize[2]);
}

int detectron2::Yolov3NmsPlugin::enqueue(int batch_size, const void * const * inputs, void ** outputs, void * workspace, cudaStream_t stream)
{
	int sum = 0;
	for (int i = 0; i < 3; i++)
	{
		if (mFeatureSize[i] * 3 < 1000)
		{
			sum = sum + mFeatureSize[i] * 3;
		}
		else
		{
			sum = sum + 1000;
		}
	}

	void* mlvl_bboxes = workspace;
	size_t mlvl_bboxes_size = floatSize(batch_size, sum * 4);

	void* mlvl_scores = nextWorkspacePtr((int8_t*)mlvl_bboxes, mlvl_bboxes_size);
	size_t mlvl_scores_size = floatSize(batch_size, sum * mParams.nbCls);

	void* mlvl_conf_scores = nextWorkspacePtr((int8_t*)mlvl_scores, mlvl_scores_size);
	size_t mlvl_conf_scores_size = floatSize(batch_size, sum * 1);

	void* next = nextWorkspacePtr((int8_t*)mlvl_conf_scores, mlvl_conf_scores_size);
	void* next1 = nextWorkspacePtr((int8_t*)mlvl_conf_scores, mlvl_conf_scores_size);

	//std::vector<int> stride = { 32, 16, 8 };

	for (int i = 0; i < 3; i++)
	{
		if (i == 0)
		{
			yolov3Detection(stream, batch_size, next, inputs[2 * i], inputs[2 * i + 1], mlvl_bboxes, mlvl_scores, mlvl_conf_scores, mParams.nbCls, i, mFeatureSize[i], mParams.stride[i], 0, sum);
		}
		else
		{
			int concatIndex = 0;
			for (int j = 0; j < i; j++)
			{
				if (mFeatureSize[j] * 3 < 1000)
				{
					concatIndex = concatIndex + mFeatureSize[j] * 3;
				}
				else
				{
					concatIndex = concatIndex + 1000;
				}
			}
			yolov3Detection(stream, batch_size, next, inputs[2 * i], inputs[2 * i + 1], mlvl_bboxes, mlvl_scores, mlvl_conf_scores, mParams.nbCls, i, mFeatureSize[i], mParams.stride[i], concatIndex, sum);
		}
	}

	//float* h_bbox = new float[3000 * 4];
	//cudaMemcpy(h_bbox, (float*)mlvl_bboxes, 3000 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
	//for (int i = 0; i < 100 * 4; i++)
	//{
	//	if (h_bbox[i] == (-1))
	//	{
	//		continue;
	//	}
	//	if (i % 4 == 0)
	//	{
	//		std::cout << "-----------------" << std::endl;
	//	}

	//	std::cout << "box:" << h_bbox[i] << std::endl;
	//}

	//clock_t start, end;
	//start = clock();
	//output[0]:预测框，output[1]:类别，output[2]:分数
	//mParams.factor_scales[0] = (float)672 / 640;
	//mParams.factor_scales[1] = (float)672 / 640;
	//mParams.factor_scales[2] = (float)672 / 640;
	//mParams.factor_scales[3] = (float)672 / 640;
	//mParams.iou_thr = 0.5;
	yolov3Nms(stream, batch_size, next1, (float*)mlvl_bboxes, (float*)mlvl_scores, (float*)mlvl_conf_scores, mParams.conf_thr, mParams.score_thr, mParams.factor_scales[0], mParams.factor_scales[1], mParams.factor_scales[2], mParams.factor_scales[3], sum, mParams.nbCls, (float*)outputs[0], (int*)outputs[1], (float*)outputs[2], mParams.iou_thr);



	return 0;
}

size_t detectron2::Yolov3NmsPlugin::getSerializationSize() const
{
	return (sizeof(int) * 7 + sizeof(float) * 7);
}

void detectron2::Yolov3NmsPlugin::serialize(void * buffer) const
{
	char *d = reinterpret_cast<char*>(buffer), *a = d;
	write(d, mParams.nbCls);
	write(d, mParams.conf_thr);
	write(d, mParams.score_thr);
	write(d, mParams.iou_thr);

	for (int i = 0; i < 4; i++)
	{
		write(d, mParams.factor_scales[i]);
	}
	for (int i = 0; i < 3; i++)
	{
		write(d, mFeatureSize[i]);
	}
	for (int i = 0; i < 3; i++)
	{
		write(d, mParams.stride[i]);
	}

	ASSERT(d == a + getSerializationSize());
}

bool detectron2::Yolov3NmsPlugin::supportsFormat(DataType type, PluginFormat format) const
{
	return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}

const char * detectron2::Yolov3NmsPlugin::getPluginType() const
{
	return "NMS_YOLOV3_TRT";
}

const char * detectron2::Yolov3NmsPlugin::getPluginVersion() const
{
	return "1";
}

IPluginV2Ext * detectron2::Yolov3NmsPlugin::clone() const
{
	IPluginV2Ext* plugin = new Yolov3NmsPlugin(mParams, mFeatureSize.data());
	return plugin;
}

void detectron2::Yolov3NmsPlugin::setPluginNamespace(const char * pluginNamespace)
{
	mPluginNamespace = pluginNamespace;
}

const char * detectron2::Yolov3NmsPlugin::getPluginNamespace() const
{
	return mPluginNamespace.c_str();
}

DataType detectron2::Yolov3NmsPlugin::getOutputDataType(int index, const nvinfer1::DataType * inputTypes, int nbInputs) const
{
	return DataType::kFLOAT;
}

bool detectron2::Yolov3NmsPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool * inputIsBroadcasted, int nbInputs) const
{
	return false;
}

bool detectron2::Yolov3NmsPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
	return false;
}

void detectron2::Yolov3NmsPlugin::configurePlugin(const Dims * inputDims, int nbInputs, const Dims * outputDims, int nbOutputs, const DataType * inputTypes, const DataType * outputTypes, const bool * inputIsBroadcast, const bool * outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
}


//----------------------------------- creator-----------------------------------

detectron2::Yolov3NmsPluginCreatorTorch::Yolov3NmsPluginCreatorTorch()
{
}

const char * detectron2::Yolov3NmsPluginCreatorTorch::getPluginName() const
{
	return NMS_YOLOV3_PLUGIN_NAME;
}

const char * detectron2::Yolov3NmsPluginCreatorTorch::getPluginVersion() const
{
	return NMS_YOLOV3_PLUGIN_VERSION;
}

const PluginFieldCollection * detectron2::Yolov3NmsPluginCreatorTorch::getFieldNames()
{
	return nullptr;
}

IPluginV2Ext * detectron2::Yolov3NmsPluginCreatorTorch::createPlugin(const char * name, const PluginFieldCollection * fc)
{
	std::cout << "creator plugin" << std::endl;
	return nullptr;
}

IPluginV2Ext * detectron2::Yolov3NmsPluginCreatorTorch::deserializePlugin(const char * name, const void * serialData, size_t serialLength)
{
	Yolov3NmsPlugin* obj = new Yolov3NmsPlugin(serialData, serialLength);
	return obj;
}

