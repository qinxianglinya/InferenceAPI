/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "nmsPluginTorch.h"
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include "bboxUtils.h"

using namespace detectronPlugin;

using namespace std;
namespace
{
	const char* NMS_TORCH_PLUGIN_VERSION{ "1" };
	const char* NMS_TORCH_PLUGIN_NAME{ "NMS_TORCH_TRT" };
} // namespace


detectronPlugin::DetectionOutput::DetectionOutput(DetectionOutputParametersTorch params, const int* featureSizeIn, const int* topkCandidateIn)
{
	mParam.nbCls = params.nbCls;
	mParam.topK = params.topK;
	mParam.keepTopK = params.keepTopK;
	mParam.nbLayer = params.nbLayer;
	mParam.nbPriorbox = params.nbPriorbox;
	mFeatureSize.resize(mParam.nbLayer);
	mTopkCandidates.resize(mParam.nbLayer);
	//mParam.featureSize.resize(mParam.nbLayer);
	//mParam.topkCandidates.resize(mParam.nbLayer);
	for (int i = 0; i < params.nbLayer; i++)
	{
		mFeatureSize[i] = featureSizeIn[i];
		mTopkCandidates[i] = topkCandidateIn[i];
	}
	mParam.srcW = params.srcW;
	mParam.srcH = params.srcH;
	mParam.targetW = params.targetW;
	mParam.targetH = params.targetH;
	mParam.scoreThreshold = params.scoreThreshold;
	mParam.iouThreshold = params.iouThreshold;
}

// Constrcutor
//DetectionOutput::DetectionOutput(int keepTopk, int topK, int srcW, int srcH, int targetW, int targetH, float scoreThreshold, float iouThreshold)
//{
//	mParam.keepTopK = keepTopk;//每层取前keepTopK个进行nms
//	mParam.topK = topK;//每张图片最多有topk个预测框
//	mScoreThreshold = scoreThreshold;
//	mIouThreshold = iouThreshold;
//	mSrcW = srcW;
//	mSrcH = srcH;
//	mTargetW = targetW;
//	mTargetH = targetH;
//}

//detectronPlugin::DetectionOutput::DetectionOutput(DetectionOutputParametersTorch params, float scoreThreshold,
//	float iouThreshold, int srcW, int srcH, int targetW, int targetH) :mParam(params), mScoreThreshold(scoreThreshold),
//	mIouThreshold(iouThreshold), mSrcW(srcW), mSrcH(srcH), mTargetW(targetW), mTargetH(targetH)
//{
//}

// Parameterized constructor
DetectionOutput::DetectionOutput(const void* data, size_t length)
{
	const char *d = reinterpret_cast<const char*>(data), *a = d;
	mParam.scoreThreshold = read<float>(d);
	mParam.iouThreshold = read<float>(d);
	mParam.srcW = read<int>(d);
	mParam.srcH = read<int>(d);
	mParam.targetW = read<int>(d);
	mParam.targetH = read<int>(d);
	mParam.nbCls = read<int>(d);
	mParam.topK = read<int>(d);
	mParam.keepTopK = read<int>(d);
	mParam.nbLayer = read<int>(d);
	mParam.nbPriorbox = read<int>(d);
	mFeatureSize.resize(mParam.nbLayer);
	mTopkCandidates.resize(mParam.nbLayer);
	for (int i = 0; i < mParam.nbLayer; i++)
	{
		mFeatureSize[i] = read<int>(d);
	}
	for (int i = 0; i < mParam.nbLayer; i++)
	{
		mTopkCandidates[i] = read<int>(d);
	}
}

int DetectionOutput::getNbOutputs() const
{
	return 3;
}

int DetectionOutput::initialize()
{
	return STATUS_SUCCESS;
}

void DetectionOutput::terminate() {}

// Returns output dimensions at given index
Dims DetectionOutput::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
	if (index == 0)
	{
		return DimsCHW(1,mParam.keepTopK*mParam.nbLayer * 4, 1);
	}
	return DimsCHW(1, mParam.keepTopK*mParam.nbLayer, 1);
}

// Returns the workspace size
size_t DetectionOutput::getWorkspaceSize(int maxBatchSize) const
{
	return detectionTorchInferenceWorkspaceSize(maxBatchSize, mParam.keepTopK, mParam.nbLayer, mParam.nbCls, mParam, mFeatureSize, mTopkCandidates);
}

// Plugin layer implementation
int DetectionOutput::enqueue(
	int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
	//std::cout << "nms enqueue" << std::endl;
	//开辟空间存放：预测框、分数、类别信息
	void* boxPtr = workspace;
	size_t boxSize = predictDataSize(batchSize, mParam.nbLayer * mParam.keepTopK * 4);

	void* scorePtr = nextWorkspacePtr((int8_t*)boxPtr, boxSize);
	size_t scoreSize = predictDataSize(batchSize, mParam.nbLayer * mParam.keepTopK);

	void* classPtr = nextWorkspacePtr((int8_t*)scorePtr, scoreSize);
	size_t classSize = indexDataSize(batchSize, mParam.nbLayer * mParam.keepTopK);

	void* next = nextWorkspacePtr((int8_t*)classPtr, classSize);
	size_t tempSize = getWorkspaceSize(batchSize);
	//std::cout << tempSize << std::endl;
	void* next2 = nextWorkspacePtr((int8_t*)classPtr, classSize);

	void* locOut = outputs[0];
	void* scoreOut = outputs[1];
	void* clsOut = outputs[2];
	void* status;

	for (int i = 0; i < mParam.nbLayer; i++)
	{
		/*void* status = detectionInferenceTorch(stream, batchSize, next, inputs[2* mParam.nbLayer + i], inputs[i], inputs[mParam.nbLayer + i], boxPtr, scorePtr, classPtr, i, mFeatureSize[i],
			mParam.nbPriorbox, mParam.nbCls, mTopkCandidates[i], mParam.nbLayer, mParam.scoreThreshold, mParam.keepTopK);*/
		status = detectionInferenceTorch(stream, batchSize, next, inputs[2 * mParam.nbLayer + i], inputs[i], inputs[mParam.nbLayer + i], boxPtr, scorePtr, classPtr, i, mFeatureSize[i],
			mParam.nbPriorbox, mParam.nbCls, mTopkCandidates[i], mParam.nbLayer, mParam.scoreThreshold, mParam.keepTopK);
	}
	
	batchNms(stream, batchSize, next2, boxPtr, scorePtr, classPtr, mParam.iouThreshold, mParam.nbCls, mParam.nbLayer, mParam.topK, mParam.srcW, mParam.srcH,
		mParam.targetW, mParam.targetH, (float*)locOut, (float*)scoreOut, (int*)clsOut, mParam.keepTopK);
	
	return 0;
}

// Returns the size of serialized parameters
size_t DetectionOutput::getSerializationSize() const
{
	return (sizeof(int)*9 + sizeof(float)*2 + sizeof(int) * 2 * mParam.nbLayer);
}

// Serialization of plugin parameters
void DetectionOutput::serialize(void* buffer) const
{
	char *d = reinterpret_cast<char*>(buffer), *a = d;
	write(d, mParam.scoreThreshold);
	write(d, mParam.iouThreshold);
	write(d, mParam.srcW);
	write(d, mParam.srcH);
	write(d, mParam.targetW);
	write(d, mParam.targetH);
	write(d, mParam.nbCls);
	write(d, mParam.topK);
	write(d, mParam.keepTopK);
	write(d, mParam.nbLayer);
	write(d, mParam.nbPriorbox);
	for (int i = 0; i < mParam.nbLayer; i++)
	{
		write(d, mFeatureSize[i]);
	}
	for (int i = 0; i < mParam.nbLayer; i++)
	{
		write(d, mTopkCandidates[i]);
	}

	ASSERT(d == a + getSerializationSize());
}

// Check if the DataType and Plugin format is supported
bool DetectionOutput::supportsFormat(DataType type, PluginFormat format) const
{
	//std::cout << "nms support format" << std::endl;
	return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}

// Get the plugin type
const char* DetectionOutput::getPluginType() const
{
	return "NMS_TORCH_TRT";
}

// Get the plugin version
const char* DetectionOutput::getPluginVersion() const
{
	return "1";
}

// Clean up
void DetectionOutput::destroy()
{
	//std::cout << "destroy" << std::endl;
	delete this;
}

// Cloning the plugin
IPluginV2Ext* DetectionOutput::clone() const
{
	// Create a new instance
	IPluginV2Ext* plugin = new DetectionOutput(mParam, mFeatureSize.data(), mTopkCandidates.data());

	// Set the namespace
	//plugin->setPluginNamespace(mPluginNamespace);
	return plugin;
}

// Set plugin namespace
void DetectionOutput::setPluginNamespace(const char* pluginNamespace)
{
	mPluginNamespace = pluginNamespace;
}

const char* DetectionOutput::getPluginNamespace() const
{
	return mPluginNamespace.c_str();
}

// Return the DataType of the plugin output at the requested index.
DataType DetectionOutput::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
	// Two outputs
	//ASSERT(index == 0 || index == 1);
	return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool DetectionOutput::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
	return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool DetectionOutput::canBroadcastInputAcrossBatch(int inputIndex) const
{
	return false;
}

// Configure the layer with input and output data types.
// inutDims: input Dimensions for the plugin layer
// nInputs : Number of inputs to the plugin layer
// outputDims: output Dimensions from the plugin layer
// nOutputs: number of outputs from the plugin layer
// type: DataType configuration for the plugin layer
// format: format NCHW, NHWC etc
// maxbatchSize: maximum batch size for the plugin layer
void DetectionOutput::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
	const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
	const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
	//std::cout << "post test---------------" << std::endl;
	//std::cout << "nbInputs:" << nbInputs << std::endl;
	//std::cout << inputDims[2].d[0] << std::endl;
	//std::cout << inputDims[2].d[1] << std::endl;
	//std::cout << inputDims[2].d[2] << std::endl;
	//std::cout << "config batchsize:" << maxBatchSize << std::endl;
	//mParam.nbLayer = (nbInputs)/3;
	//mParam.nbPriorbox = inputDims[mParam.nbLayer].d[0] / 4;
	//mParam.nbCls = inputDims[0].d[0] / mParam.nbPriorbox;
	//for (int i = 0; i < mParam.nbLayer; i++)
	//{
	//	int featureSize = inputDims[i].d[1] * inputDims[i].d[2];
	//	mParam.featureSize.push_back(featureSize);
	//	int keep = std::min(featureSize*mParam.nbPriorbox, mParam.keepTopK);
	//	mParam.topkCandidates.push_back(keep);
	//}
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void DetectionOutput::attachToContext(
	cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void DetectionOutput::detachFromContext() {}

detectronPlugin::DetectionOutputCreator::DetectionOutputCreator()
{
	//std::cout << "nms creator" << std::endl;
}

const char * detectronPlugin::DetectionOutputCreator::getPluginName() const
{
	//std::cout << "nms creator getplugin name" << std::endl;
	return NMS_TORCH_PLUGIN_NAME;
}

const char * detectronPlugin::DetectionOutputCreator::getPluginVersion() const
{
	//std::cout << "nms creator getplugin version" << std::endl;
	return NMS_TORCH_PLUGIN_VERSION;
}

const PluginFieldCollection * detectronPlugin::DetectionOutputCreator::getFieldNames()
{
	return nullptr;
}

IPluginV2Ext * detectronPlugin::DetectionOutputCreator::createPlugin(const char * name, const PluginFieldCollection * fc)
{
	return nullptr;
}

IPluginV2Ext * detectronPlugin::DetectionOutputCreator::deserializePlugin(const char * name, const void * data, size_t length)
{
	//std::cout << "nms creator deserialize" << std::endl;
	DetectionOutput* obj = new DetectionOutput(data, length);
	return obj;
}
