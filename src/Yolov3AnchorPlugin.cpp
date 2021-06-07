#include "Yolov3AnchorPlugin.h"
//#include "struct.h"
//using namespace detectron2;
//using namespace myPlugin;
namespace
{
	const char* ANCHOR_YOLOV3_PLUGIN_VERSION{ "1" };
	const char* ANCHOR_YOLOV3_PLUGIN_NAME{ "ANCHOR_YOLOV3_TRT" };
} // namespace

detectron2::Yolov3AnchorPlugin::Yolov3AnchorPlugin(const AnchorParamsYolov3* paramsIn, int numLayer)
{
	mLayerNum = numLayer;
	mParams.resize(mLayerNum);
	for (int i = 0; i < mLayerNum; i++)
	{
		mParams[i] = paramsIn[i];
		mParams[i].nStride = paramsIn[i].nStride;
		for (int j = 0; j < 3; j++)
		{
			mParams[i].fBaseAnchor[j] = paramsIn[i].fBaseAnchor[j];
		}
		mParams[i].featureSize = paramsIn[i].featureSize;
	}
}

detectron2::Yolov3AnchorPlugin::Yolov3AnchorPlugin(const void* data, size_t length)
{
	const char *d = reinterpret_cast<const char*>(data), *a = d;
	mLayerNum = read<int>(d);
	mParams.resize(mLayerNum);
	for (int id = 0; id < mLayerNum; id++)
	{
		// we have to deserialize GridAnchorParameters by hand
		mParams[id].nStride = read<int>(d);

		for (int i = 0; i < 3; i++)
		{
			mParams[id].fBaseAnchor[i] = read<Coordinate>(d);
			mParams[id].fBaseSize[i] = read<Coordinate2d>(d);
		}
		mParams[id].featureSize = read<FeatureSize>(d);
	}
	ASSERT(d == a + length);
}

int detectron2::Yolov3AnchorPlugin::getNbOutputs() const
{
	return mParams.size();
}

Dims detectron2::Yolov3AnchorPlugin::getOutputDimensions(int index, const Dims * inputs, int nbInputDims)
{
	return DimsCHW(mParams[index].featureSize.nWidth*mParams[index].featureSize.nHeight * 3 * 4, 1, 1);
}

int detectron2::Yolov3AnchorPlugin::initialize()
{
	return 0;
}

void detectron2::Yolov3AnchorPlugin::terminate()
{
}

void detectron2::Yolov3AnchorPlugin::destroy()
{
	delete this;
}

size_t detectron2::Yolov3AnchorPlugin::getWorkspaceSize(int) const
{
	return size_t();
}

int detectron2::Yolov3AnchorPlugin::enqueue(int batch_size, const void * const * inputs, void ** outputs, void * workspace, cudaStream_t stream)
{
	for (int i = 0; i < mLayerNum; i++)
	{
		void *outputData = outputs[i];
		pluginStatus_t status = yolov3Anchor(stream, mParams[i], outputData);
		//if (i == 0)
		//{
		//	float *hOutput = new float[4800];
		//	cudaMemcpy(hOutput, outputData, 4800  * sizeof(float), cudaMemcpyDeviceToHost);
		//	for (int j = 0; j < 72; j++)
		//	{
		//		if (j % 12 == 0)
		//			std::cout << "----------" << std::endl;
		//		std::cout << "anchor:" << hOutput[j] << std::endl;
		//	}
		//}
	}



	return 0;
}

size_t detectron2::Yolov3AnchorPlugin::getSerializationSize() const
{
	size_t sum = sizeof(int); // mNumLayers
	for (int i = 0; i < mLayerNum; i++)
	{
		sum += sizeof(int);
		sum += sizeof(Coordinate) * 3;
		sum += sizeof(Coordinate2d) * 3; 
		sum += sizeof(FeatureSize);
	}
	return sum;
}

void detectron2::Yolov3AnchorPlugin::serialize(void * buffer) const
{
	char *d = reinterpret_cast<char*>(buffer), *a = d;
	write(d, mLayerNum);
	for (int id = 0; id < mParams.size(); id++)
	{
		// we have to serialize GridAnchorParameters by hand
		write(d, mParams[id].nStride);
		for (int i = 0; i < 3; i++)
		{
			write(d, mParams[id].fBaseAnchor[i]);
			write(d, mParams[id].fBaseSize[i]);
		}
		write(d, mParams[id].featureSize);
	}
	ASSERT(d == a + getSerializationSize());
}

bool detectron2::Yolov3AnchorPlugin::supportsFormat(DataType type, PluginFormat format) const
{
	return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}

const char * detectron2::Yolov3AnchorPlugin::getPluginType() const
{
	return ANCHOR_YOLOV3_PLUGIN_NAME;
}

const char * detectron2::Yolov3AnchorPlugin::getPluginVersion() const
{
	return ANCHOR_YOLOV3_PLUGIN_VERSION;
}

IPluginV2Ext * detectron2::Yolov3AnchorPlugin::clone() const
{
	IPluginV2Ext* plugin = new Yolov3AnchorPlugin(mParams.data(), mLayerNum);
	return plugin;
}

void detectron2::Yolov3AnchorPlugin::setPluginNamespace(const char * libNamespace)
{
	mNameSpace = libNamespace;
}

const char * detectron2::Yolov3AnchorPlugin::getPluginNamespace() const
{
	return mNameSpace.c_str();
}

DataType detectron2::Yolov3AnchorPlugin::getOutputDataType(int index, const nvinfer1::DataType * inputTypes, int nbInputs) const
{
	return DataType::kFLOAT;;
}

bool detectron2::Yolov3AnchorPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool * inputIsBroadcasted, int nbInputs) const
{
	return false;
}

bool detectron2::Yolov3AnchorPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
	return false;
}

void detectron2::Yolov3AnchorPlugin::configurePlugin(const Dims * inputDims, int nbInputs, const Dims * outputDims, int nbOutputs, const DataType * inputTypes, const DataType * outputTypes, const bool * inputIsBroadcast, const bool * outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
}

//plugin creator
detectron2::Yolov3AnchorPluginCreatorTorch::Yolov3AnchorPluginCreatorTorch()
{
}

const char * detectron2::Yolov3AnchorPluginCreatorTorch::getPluginName() const
{
	return ANCHOR_YOLOV3_PLUGIN_NAME;
}

const char * detectron2::Yolov3AnchorPluginCreatorTorch::getPluginVersion() const
{
	return ANCHOR_YOLOV3_PLUGIN_VERSION;
}

const PluginFieldCollection * detectron2::Yolov3AnchorPluginCreatorTorch::getFieldNames()
{
	return nullptr;
}

IPluginV2Ext * detectron2::Yolov3AnchorPluginCreatorTorch::createPlugin(const char * name, const PluginFieldCollection * fc)
{
	return nullptr;
}

IPluginV2Ext * detectron2::Yolov3AnchorPluginCreatorTorch::deserializePlugin(const char * name, const void * serialData, size_t serialLength)
{
	Yolov3AnchorPlugin* obj = new Yolov3AnchorPlugin(serialData, serialLength);
	return obj;
}
