#include "anchorGeneratorTorch.h"
#include <math.h>

//base size:初始框的大小-> anchor generator size
//
//
namespace
{
	const char* ANCHOR_TORCH_PLUGIN_VERSION{ "1" };
	const char* ANCHOR_TORCH_PLUGIN_NAME{ "ANCHOR_TORCH_TRT" };
} // namespace

detectronPlugin::AnchorGeneratorTorch::AnchorGeneratorTorch(const AnchorParamsTorch* paramsIn, int numLayer)
{
	//
	//std::cout << "constructor" << std::endl;
	mLayerNum = numLayer;
	mParams.resize(mLayerNum);
	for (int i = 0; i < mLayerNum; i++)
	{
		mParams[i] = paramsIn[i];
		mParams[i].nOffset = paramsIn[i].nOffset;
		mParams[i].nStride = paramsIn[i].nStride;
		for (int j = 0; j < 3; j++)
		{
			mParams[i].nSize[j] = paramsIn[i].nSize[j];
			mParams[i].fAspectRatio[j] = paramsIn[i].fAspectRatio[j];
		}
		for (int j = 0; j < 9; j++)
		{
			mParams[i].fBaseAnchor[j] = paramsIn[i].fBaseAnchor[j];
		}
		mParams[i].featureSize = paramsIn[i].featureSize;
	}
}

detectronPlugin::AnchorGeneratorTorch::AnchorGeneratorTorch(const void * data, size_t length)
{
	const char *d = reinterpret_cast<const char*>(data), *a = d;
	mLayerNum = read<int>(d);
	mParams.resize(mLayerNum);
	//CUASSERT(cudaMallocHost((void**)&mNumPriors, 1 * sizeof(int)));
	//CUASSERT(cudaMallocHost((void**)&mDeviceWidths, 1 * sizeof(Weights)));
	//CUASSERT(cudaMallocHost((void**)&mDeviceHeights, 1 * sizeof(Weights)));
	for (int id = 0; id < mLayerNum; id++)
	{
		// we have to deserialize GridAnchorParameters by hand

		mParams[id].nOffset = read<int>(d);
		mParams[id].nStride = read<int>(d);
		for (int i = 0; i < 3; i++)
		{
			mParams[id].nSize[i] = read<float>(d);
		}
		for (int i = 0; i < 3; i++)
		{
			mParams[id].fAspectRatio[i] = read<float>(d);
		}
		for (int i = 0; i < 9; i++)
		{
			mParams[id].fBaseAnchor[i] = read<Coordinate>(d);
		}
		mParams[id].featureSize = read<FeatureSize>(d);
	}
	ASSERT(d == a + length);
}

int detectronPlugin::AnchorGeneratorTorch::getNbOutputs() const
{
	return mParams.size();
}
//index控制输出维度
Dims detectronPlugin::AnchorGeneratorTorch::getOutputDimensions(int index, const Dims * inputs, int nbInputDims)
{
	return DimsCHW(mParams[index].featureSize.nWidth*mParams[index].featureSize.nHeight * 9 * 4, 1, 1);
	//std::cout << "anchor getoutDim" << std::endl;
	//if (index == 0)
	//{
	//	return DimsCHW(360000, 1, 1);
	//}
	//else if (index == 1)
	//{
	//	return DimsCHW(22500*4, 1, 1);
	//}
	//else if (index == 2)
	//{
	//	return DimsCHW(5625*4, 1,  1);
	//}
	//else if (index == 3)
	//{
	//	return DimsCHW(1521*4, 1, 1);
	//}
	//else if(index == 4)
	//{
	//	return DimsCHW(441*4, 1,  1);
	//}
}

int detectronPlugin::AnchorGeneratorTorch::initialize()
{
	//std::cout << "anchor initialize" << std::endl;
	return STATUS_SUCCESS;
}

void detectronPlugin::AnchorGeneratorTorch::terminate()
{
	//std::cout << "anchor terminate" << std::endl;
}

void detectronPlugin::AnchorGeneratorTorch::destroy()
{
	//std::cout << "anchor destroy" << std::endl;
	delete this;
}

size_t detectronPlugin::AnchorGeneratorTorch::getWorkspaceSize(int) const
{
	//std::cout << "anchor getWorkspaceSize" << std::endl;
	return 0;
}

int detectronPlugin::AnchorGeneratorTorch::enqueue(int batch_size, const void * const * inputs, void ** outputs, void * workspace, cudaStream_t stream)
{
	//std::cout << "anchor enqueue" << std::endl;
	for (int i = 0; i < mLayerNum; i++)
	{
		//std::cout << "enqueue" << std::endl;
		//std::cout << "size:" << mParams.size() << std::endl;
		void *outputData = outputs[i];
		pluginStatus_t status =  anchorGridTorch(stream, mParams[i], outputData);
		//if (i==0)
		//{
		//	float *hOutput = new float[360000*4];
		//	cudaMemcpy(hOutput, outputData, 360000*4 *  sizeof(float), cudaMemcpyDeviceToHost);
		//	for (int j = 0; j < 72; j++)
		//	{
		//		if (j % 36 == 0)
		//			std::cout << "----------" << std::endl;
		//		std::cout << "anchor:" << hOutput[j] << std::endl;
		//	}
		//	//std::cout << "anchor:" << hOutput[298132] << std::endl;
		//	//std::cout << "anchor:" << hOutput[298133] << std::endl;
		//	//std::cout << "anchor:" << hOutput[298134] << std::endl;
		//	//std::cout << "anchor:" << hOutput[298135] << std::endl;
		//}
	}


	return STATUS_SUCCESS;
}

//int nOffset;
//int nStride;
//float nSize[3];
//float fAspectRatio[3] = { 0.5, 1, 2 };
//Coordinate fBaseAnchor[9];
//FeatureSize featureSize;
//float anchorScale;
size_t detectronPlugin::AnchorGeneratorTorch::getSerializationSize() const
{
	//std::cout << "anchor getSerializationSize" << std::endl;
	size_t sum = sizeof(int); // mNumLayers
	for (int i = 0; i < mLayerNum; i++)
	{
		sum += 2* sizeof(int); 
		sum += sizeof(FeatureSize);
		sum += sizeof(Coordinate)*9; // mNumPriors, mParam[i].{numAspectRatios, H, W}
		sum += 6 * sizeof(float); // mParam[i].{minSize, maxSize, aspectRatios, variance[4]}
	}
	return sum;
}



void detectronPlugin::AnchorGeneratorTorch::serialize(void * buffer) const
{
	//std::cout << "anchor serialize" << std::endl;
	char *d = reinterpret_cast<char*>(buffer), *a = d;
	write(d, mLayerNum);
	for (int id = 0; id < mParams.size(); id++)
	{
		// we have to serialize GridAnchorParameters by hand
		write(d, mParams[id].nOffset);
		write(d, mParams[id].nStride);
		for (int i = 0; i < 3; i++)
		{
			write(d, mParams[id].nSize[i]);
		}
		for (int i = 0; i < 3; i++)
		{
			write(d, mParams[id].fAspectRatio[i]);
		}
		for (int i = 0; i < 9; i++)
		{
			write(d, mParams[id].fBaseAnchor[i]);
		}
		write(d, mParams[id].featureSize);
	}
	ASSERT(d == a + getSerializationSize());
}

bool detectronPlugin::AnchorGeneratorTorch::supportsFormat(DataType type, PluginFormat format) const
{
	//std::cout << "anchor supportsFormat" << std::endl;
	return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);;
}

const char * detectronPlugin::AnchorGeneratorTorch::getPluginType() const
{
	//std::cout << "anchor getPluginType" << std::endl;
	return ANCHOR_TORCH_PLUGIN_NAME;
}

const char * detectronPlugin::AnchorGeneratorTorch::getPluginVersion() const
{
	//std::cout << "anchor getPluginVersion" << std::endl;
	return ANCHOR_TORCH_PLUGIN_VERSION;
}

IPluginV2Ext * detectronPlugin::AnchorGeneratorTorch::clone() const
{
	//std::cout << "anchor clone" << std::endl;
	IPluginV2Ext* plugin = new AnchorGeneratorTorch(mParams.data(), mLayerNum);
	//plugin->setPluginNamespace(ANCHOR_PLUGIN_NAME);
	return plugin;
}

void detectronPlugin::AnchorGeneratorTorch::setPluginNamespace(const char * libNamespace)
{
	//std::cout << "anchor setPluginNamespace" << std::endl;
	mNameSpace = libNamespace;
}

const char * detectronPlugin::AnchorGeneratorTorch::getPluginNamespace() const
{
	//std::cout << "anchor getPluginNamespace" << std::endl;
	return mNameSpace.c_str();
}

DataType detectronPlugin::AnchorGeneratorTorch::getOutputDataType(int index, const nvinfer1::DataType * inputTypes, int nbInputs) const
{
	//std::cout << "anchor getOutputDataType" << std::endl;
	return DataType::kFLOAT;;
}

bool detectronPlugin::AnchorGeneratorTorch::isOutputBroadcastAcrossBatch(int outputIndex, const bool * inputIsBroadcasted, int nbInputs) const
{
	return false;
}

bool detectronPlugin::AnchorGeneratorTorch::canBroadcastInputAcrossBatch(int inputIndex) const
{
	return false;
}

void detectronPlugin::AnchorGeneratorTorch::configurePlugin(const Dims * inputDims, int nbInputs, const Dims * outputDims, int nbOutputs, const DataType * inputTypes, const DataType * outputTypes, const bool * inputIsBroadcast, const bool * outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
}

detectronPlugin::GridAnchorBasePluginCreatorTorch::GridAnchorBasePluginCreatorTorch()
{
	//mNamespace = ANCHOR_TORCH_PLUGIN_NAME;
}

const char * detectronPlugin::GridAnchorBasePluginCreatorTorch::getPluginName() const
{
	return ANCHOR_TORCH_PLUGIN_NAME;
}

const char * detectronPlugin::GridAnchorBasePluginCreatorTorch::getPluginVersion() const
{
	return ANCHOR_TORCH_PLUGIN_VERSION;
}

const PluginFieldCollection * detectronPlugin::GridAnchorBasePluginCreatorTorch::getFieldNames()
{
	return nullptr;
}

IPluginV2Ext * detectronPlugin::GridAnchorBasePluginCreatorTorch::createPlugin(const char * name, const PluginFieldCollection * fc)
{
	//std::cout << "GridAnchorBasePluginCreatorTorch create Plugin" << std::endl;
	return nullptr;
}

IPluginV2Ext * detectronPlugin::GridAnchorBasePluginCreatorTorch::deserializePlugin(const char * name, const void * serialData, size_t serialLength)
{
	AnchorGeneratorTorch* obj = new AnchorGeneratorTorch(serialData, serialLength);

	//obj->setPluginNamespace(mNamespace.c_str());
	return obj;
}

//detectronPlugin::GridAnchorPluginCreatorTorch::GridAnchorPluginCreatorTorch()
//{
//	mPluginName = ANCHOR_PLUGIN_NAME;
//}
