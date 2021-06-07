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
#include "gridAnchorPlugin.h"
#include <cstring>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <math.h>

using namespace nvinfer1;
using namespace std;

namespace
{
	const char* GRID_ANCHOR_PLUGIN_NAMES[] = { "GridAnchor_TRT", "GridAnchorRect_TRT" };
	const char* GRID_ANCHOR_PLUGIN_VERSION = "1";
} // namespace

PluginFieldCollection GridAnchorBasePluginCreator::mFC{};
std::vector<PluginField> GridAnchorBasePluginCreator::mPluginAttributes;
// boxParams[i] = {minScale, maxScale, aspectRatios.data(), (int) aspectRatios.size(), 
//fMapShapes[hOffset],fMapShapes[wOffset], {layerVariances[0],        
//layerVariances[1], layerVariances[2], layerVariances[3]}};
//numLayers, mPluginName
GridAnchorGenerator::GridAnchorGenerator(const GridAnchorParametersTf* paramIn, int anchorSca, int numLayers, const char *name)
	: mNumLayers(numLayers), mPluginName(name)
{
	CUASSERT(cudaMallocHost((void**)&mNumPriors, mNumLayers * sizeof(int)));
	CUASSERT(cudaMallocHost((void**)&mDeviceWidths, mNumLayers * sizeof(Weights)));
	CUASSERT(cudaMallocHost((void**)&mDeviceHeights, mNumLayers * sizeof(Weights)));

	mParam.resize(1);
	for (int id = 0; id < 1; id++)
	{
		mParam[id] = paramIn[id];
		// ASSERT(mParam[id].numAspectRatios >= 0 && mParam[id].aspectRatios != nullptr);

		mParam[id].aspectRatios = (float*)malloc(sizeof(float) * mParam[id].numAspectRatios);
		//初始化aspectRatios
		for (int i = 0; i < paramIn[id].numAspectRatios; ++i)
		{
			mParam[id].aspectRatios[i] = paramIn[id].aspectRatios[i];
		}
		//初始化特征图宽高
		mParam[id].W = paramIn[id].W;
		mParam[id].H = paramIn[id].H;
		//初始化variance
		for (int i = 0; i < 4; ++i)
		{
			mParam[id].variance[i] = paramIn[id].variance[i];
		}

		mParam[id].scalesPerOctave = paramIn[id].scalesPerOctave;
		//        cout<<"octave:"<<mParam[id].scalesPerOctave<<endl;

				//计算单位像素的锚框个数
		mNumPriors[id] = mParam[id].scalesPerOctave * mParam[id].numAspectRatios;

		// 初始化level
		mParam[id].level = paramIn[id].level;

		//初始化anchorScale
		anchorScale = anchorSca;

		//根据level和anchor_scale计算出baseSize
		int baseSize = anchorScale * pow(2, mParam[id].level);
		//        cout<<"baseSize:"<<baseSize<<endl;

				//初始化scales
		vector<float> scales(mParam[id].scalesPerOctave);

		for (int i = 0; i < mParam[id].scalesPerOctave; i++)
		{
			float k = (float)i / (mParam[id].scalesPerOctave);
			//            cout<<"k:"<<k<<endl;
			float s = pow(2, k);
			scales[i] = s;
			//            cout<<"scales"<<s<<endl;
		}

		//根据baseSize,scales和aspectRatios计算出宽高
		std::vector<float> tmpWidths;
		std::vector<float> tmpHeights;
		// Calculate the width and height of the prior boxes
		// for (int i = 0; i < mParam[id].scalesPerOctave; i++)
		// {
		//     for(int j=0; j < mParam[id].numAspectRatios; j++)
		//     {
		//         float sqrt_AR = sqrt(mParam[id].aspectRatios[j]);
		//         tmpWidths.push_back(baseSize*scales[i] * sqrt_AR);
		//         cout<<"w:"<<baseSize*scales[i] * sqrt_AR<<endl;
		//         tmpHeights.push_back(baseSize*scales[i] / sqrt_AR);
		//         cout<<"h:"<<baseSize*scales[i] / sqrt_AR<<endl;
		//     }
		// }
		for (int i = 0; i < mParam[id].numAspectRatios; i++)
		{
			for (int j = 0; j < mParam[id].scalesPerOctave; j++)
			{
				float sqrt_AR = sqrt(mParam[id].aspectRatios[i]);
				tmpWidths.push_back(baseSize*scales[j] * sqrt_AR);
				tmpHeights.push_back(baseSize*scales[j] / sqrt_AR);
			}
		}

		mDeviceWidths[id] = copyToDevice(&tmpWidths[0], tmpWidths.size());
		mDeviceHeights[id] = copyToDevice(&tmpHeights[0], tmpHeights.size());
	}
	//    cout<<"grid construct endl"<<endl;
}
// stream, mParam[id], mNumPriors[id], mDeviceWidths[id].values, 
// mDeviceHeights[id].values, outputData
GridAnchorGenerator::GridAnchorGenerator(const void* data, size_t length, const char *name) :
	mPluginName(name)
{
	//cout << "GridAnchorGenerator" << endl;
	const char *d = reinterpret_cast<const char*>(data), *a = d;
	mNumLayers = read<int>(d);
	CUASSERT(cudaMallocHost((void**)&mNumPriors, 1 * sizeof(int)));
	CUASSERT(cudaMallocHost((void**)&mDeviceWidths, 1 * sizeof(Weights)));
	CUASSERT(cudaMallocHost((void**)&mDeviceHeights, 1 * sizeof(Weights)));
	mParam.resize(1);
	for (int id = 0; id < 1; id++)
	{
		// we have to deserialize GridAnchorParameters by hand
		mParam[id].level = read<int>(d);
		//cout << "!!!!!!!!level:" << mParam[id].level << endl;
		mParam[id].scalesPerOctave = read<int>(d);
		mParam[id].numAspectRatios = read<int>(d);
		mParam[id].aspectRatios = (float*)malloc(sizeof(float) * mParam[id].numAspectRatios);
		for (int i = 0; i < mParam[id].numAspectRatios; ++i)
		{
			mParam[id].aspectRatios[i] = read<float>(d);
		}
		mParam[id].H = read<int>(d);
		mParam[id].W = read<int>(d);
		mParam[id].imgH = read<int>(d);
		mParam[id].imgW = read<int>(d);
		for (int i = 0; i < 4; ++i)
		{
			mParam[id].variance[i] = read<float>(d);
		}

		mNumPriors[id] = read<int>(d);
		//cout << "deserializeToDevice1 start!!!" << endl;
		mDeviceWidths[id] = deserializeToDevice(d, mNumPriors[id]);

		//cout << "deserializeToDevice2 start!!!" << endl;
		mDeviceHeights[id] = deserializeToDevice(d, mNumPriors[id]);
		anchorScale = read<int>(d);
		//cout << "???" << endl;
		//float * wr = new float[6];
		//cudaMemcpy(wr, mDeviceWidths[id].values, 6 * sizeof(float), cudaMemcpyDeviceToHost);
		//for (int i = 0; i < 6; i++)
		//{
		//	cout << "kkkkkkkkkkkkkkk" << endl;
		//	cout << wr[i] << endl;
		//}
	}
	// cout<<"minisize:"<<mParam[0].minSize<<endl;
	// cout<<"maxSize:"<<mParam[0].maxSize<<endl;
	// cout<<"numAspectRatios:"<<mParam[0].numAspectRatios<<endl;
	// for(auto aspectRatio : mParam[0].aspectRatios)
	// {
	//     cout<<"aspectRatio:"<<aspectRatio<<endl;
	// }
	// cout<<"H:"<<mParam[0].H<<endl;
	// cout<<"W:"<<mParam[0].W<<endl;
	// for(auto var :mParam[0].variance)
	// {
	//     cout<<"variance:"<<var<<endl;
	// }
	// cout<<"nbPriors:"<<mNumPriors[0]<<endl;
	// cout<<"mDeviceWidths"<<mDeviceWidths[0]<<endl;
	// cout<<"mDeviceHeights"<<mDeviceHeights[0]<<endl;

	ASSERT(d == a + length);
}

GridAnchorGenerator::~GridAnchorGenerator()
{
	for (int id = 0; id < mNumLayers; id++)
	{
		//cout << "free" << endl;
		CUERRORMSG(cudaFree(const_cast<void*>(mDeviceWidths[id].values)));
		CUERRORMSG(cudaFree(const_cast<void*>(mDeviceHeights[id].values)));
		free(mParam[id].aspectRatios);
	}
	CUERRORMSG(cudaFreeHost(mNumPriors));
	CUERRORMSG(cudaFreeHost(mDeviceWidths));
	CUERRORMSG(cudaFreeHost(mDeviceHeights));
}

int GridAnchorGenerator::getNbOutputs() const
{
	//    cout<<"grid getNbOutputs"<<endl;
	return 1;
}

Dims GridAnchorGenerator::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
	// Particularity of the PriorBox layer: no batchSize dimension needed
	// 2 channels. First channel stores the mean of each prior coordinate.
	// Second channel stores the variance of each prior coordinate.
//    cout<<"---------------------------"<<endl;
//    cout<<mParam[index].H<<"---"<<mParam[index].W<<"---"<<mNumPriors[index]<<endl;
//    cout<<"grid getOutputDimensions"<<mParam[index].H * mParam[index].W * mNumPriors[index] * 4<<endl;
	return DimsCHW(2, mParam[index].H * mParam[index].W * mNumPriors[index] * 4, 1);
}

int GridAnchorGenerator::initialize()
{
	return STATUS_SUCCESS;
}

void GridAnchorGenerator::terminate() {}

size_t GridAnchorGenerator::getWorkspaceSize(int maxBatchSize) const
{
	return 0;
}

int GridAnchorGenerator::enqueue(
	int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{

	//float * result = new float[880];
	//int baseSize_1 = anchorScale * pow(2, mParam[0].level);


	//		//初始化scales
	//vector<float> scales(mParam[0].scalesPerOctave);

	//for (int i = 0; i < mParam[0].scalesPerOctave; i++)
	//{
	//	float k = (float)i / (mParam[0].scalesPerOctave);
	//	//            cout<<"k:"<<k<<endl;
	//	float s = pow(2, k);
	//	scales[i] = s;
	//	//            cout<<"scales"<<s<<endl;
	//}
	//std::vector<float> tmpWidths;
	//std::vector<float> tmpHeights;
	//cout << "-----------" << endl;
	//for (int i = 0; i < mParam[0].numAspectRatios; i++)
	//{
	//	for (int j = 0; j < mParam[0].scalesPerOctave; j++)
	//	{
	//		float sqrt_AR = sqrt(mParam[0].aspectRatios[i]);
	//		tmpWidths.push_back(baseSize_1*scales[j] * sqrt_AR);
	//		tmpHeights.push_back(baseSize_1*scales[j] / sqrt_AR);
	//		cout << "width:" << baseSize_1 * scales[j] * sqrt_AR << endl;
	//		cout << "height:" << baseSize_1 * scales[j] / sqrt_AR << endl;
	//	}
	//}
	//cout << "---------" << endl;

	//mDeviceWidths[0] = copyToDevice(&tmpWidths[0], tmpWidths.size());
	//mDeviceHeights[0] = copyToDevice(&tmpHeights[0], tmpHeights.size());


	//输出weight
	//float* weightCpu = new float[6];

	//std::cout << "test" << std::endl;
	for (int id = 0; id < mNumLayers; id++)
	{
		void* outputData = outputs[id];
		//mParam参数，锚框的个数，锚框的宽高
//        for(int i=0; i<24; i++)
//        {
//           cout<<"mDeviceHeights[id].values:"<<mDeviceHeights[id].values[i]<<endl;
//           cout<<"mDeviceWidths[id].values:"<<mDeviceWidths[id].values[i]<<endl;
//        }

//        cout<<"level:"<<mParam[id].level<<endl;
        //cout<<"w:"<<mParam[id].W<<endl;
        //cout<<"h:"<<mParam[id].H<<endl;
		pluginStatus_t status = anchorGridInference(
			stream, mParam[id], mNumPriors[id], mDeviceWidths[id].values, mDeviceHeights[id].values, outputData);
		//        ASSERT(status == STATUS_SUCCESS);
		//        cout<<"----------------------------"<<endl;
		//if (id == 0)
		//{
		//cudaMemcpy(result, outputData, 880 * sizeof(float), cudaMemcpyDeviceToHost);
		//cudaMemcpy(weightCpu, mDeviceWidths[id].values, 6 * sizeof(float), cudaMemcpyDeviceToHost);
		//}
	}
	//cout << "-----weitht------" << endl;
	//for (int i = 0; i < 6; i++)
	//{
	//	cout << weightCpu[i] << endl;
	//	//cout << result[i] << endl;
	//}
	//cout << "-----weitht------" << endl;
	////cout << "-----result------" << endl;
	//cout << "-----result------" << endl;
	//for (int i = 0; i < 9; i++)
	//{
	//	//cout << weightCpu[i] << endl;
	//	cout << result[i] << endl;
	//}
	//cout << "-----result------" << endl;
	return STATUS_SUCCESS;
}

size_t GridAnchorGenerator::getSerializationSize() const
{
	//cout << "getSerializationSize" << endl;
	size_t sum = sizeof(int); // mNumLayers
	for (int i = 0; i < mNumLayers; i++)
	{
		sum += 8 * sizeof(int); // mNumPriors, mParam[i].{numAspectRatios, H, W}
		sum += (3 + 4)
			* sizeof(float); // mParam[i].{minSize, maxSize, aspectRatios, variance[4]}
		sum += mDeviceWidths[i].count * sizeof(float);
		sum += mDeviceHeights[i].count * sizeof(float);
	}
	//cout << "return getSerializationSize" << endl;
	sum += sizeof(int);
	return sum;
}

void GridAnchorGenerator::serialize(void* buffer) const
{
	//cout << "serialize-------------" << endl;
	char *d = reinterpret_cast<char*>(buffer), *a = d;
	write(d, mNumLayers);
	for (int id = 0; id < 1; id++)
	{
		// we have to serialize GridAnchorParameters by hand
		write(d, mParam[id].level);
		write(d, mParam[id].scalesPerOctave);
		write(d, mParam[id].numAspectRatios);
		for (int i = 0; i < 3; ++i)
		{
			write(d, mParam[id].aspectRatios[i]);
		}
		write(d, mParam[id].H);
		write(d, mParam[id].W);
		write(d, mParam[id].imgH);
		write(d, mParam[id].imgW);
		for (int i = 0; i < 4; ++i)
		{
			write(d, mParam[id].variance[i]);
		}

		write(d, mNumPriors[id]);
		serializeFromDevice(d, mDeviceWidths[id]);

		serializeFromDevice(d, mDeviceHeights[id]);
		//float * wr = new float[4];
		//cudaMemcpy(wr, mDeviceWidths[id].values, 6 * sizeof(float), cudaMemcpyDeviceToHost);
		//for (int i = 0; i < 4; i++)
		//{
		//	cout << wr[i] << endl;
		//}
		////cout << "serialize end-----------" << endl;
		write(d, anchorScale);
	}
	ASSERT(d == a + getSerializationSize());
}

Weights GridAnchorGenerator::copyToDevice(const void* hostData, size_t count)
{
	//    cout<<"copyToDevice"<<endl;
	void* deviceData;
	CUASSERT(cudaMalloc(&deviceData, count * sizeof(float)));
	CUASSERT(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
	return Weights{ DataType::kFLOAT, deviceData, int64_t(count) };
}

void GridAnchorGenerator::serializeFromDevice(char*& hostBuffer, Weights deviceWeights) const
{
	//cout << "serializeFromDevice" << endl;
	cudaMemcpy(hostBuffer, deviceWeights.values, deviceWeights.count * sizeof(float), cudaMemcpyDeviceToHost);
	hostBuffer += deviceWeights.count * sizeof(float);
}

Weights GridAnchorGenerator::deserializeToDevice(const char*& hostBuffer, size_t count)
{
	//cout << "deserializeToDevice" << endl;
	Weights w = copyToDevice(hostBuffer, count);
	hostBuffer += count * sizeof(float);
	return w;
}
bool GridAnchorGenerator::supportsFormat(DataType type, PluginFormat format) const
{
	return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}

const char* GridAnchorGenerator::getPluginType() const
{
	return mPluginName;
}

const char* GridAnchorGenerator::getPluginVersion() const
{
	return GRID_ANCHOR_PLUGIN_VERSION;
}


// Set plugin namespace
void GridAnchorGenerator::setPluginNamespace(const char* pluginNamespace)
{
	mPluginNamespace = pluginNamespace;
}

const char* GridAnchorGenerator::getPluginNamespace() const
{
	return mPluginNamespace;
}

// Return the DataType of the plugin output at the requested index
DataType GridAnchorGenerator::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
	// ASSERT(index < mNumLayers);
//    cout<<"grid getOutputDataType"<<endl;
	return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool GridAnchorGenerator::isOutputBroadcastAcrossBatch(
	int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
	return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool GridAnchorGenerator::canBroadcastInputAcrossBatch(int inputIndex) const
{
	return false;
}

// Configure the layer with input and output data types.
void GridAnchorGenerator::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
	const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
	const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
	// ASSERT(nbOutputs == mNumLayers);
	ASSERT(outputDims[0].nbDims == 3);
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void GridAnchorGenerator::attachToContext(
	cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void GridAnchorGenerator::detachFromContext() {}

void GridAnchorGenerator::destroy()
{
	delete this;
}

IPluginV2Ext* GridAnchorGenerator::clone() const
{
	//    cout<<"clone"<<endl;
	IPluginV2Ext* plugin = new GridAnchorGenerator(mParam.data(), anchorScale, mNumLayers, mPluginName);
	plugin->setPluginNamespace(mPluginNamespace);
	return plugin;
}

GridAnchorBasePluginCreator::GridAnchorBasePluginCreator()
{
	mPluginAttributes.emplace_back(PluginField("level", nullptr, PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(PluginField("anchorScale", nullptr, PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(PluginField("scalesPerOctave", nullptr, PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(PluginField("variance", nullptr, PluginFieldType::kFLOAT32, 4));
	mPluginAttributes.emplace_back(PluginField("anchorRatio", nullptr, PluginFieldType::kFLOAT32, 1));
	mPluginAttributes.emplace_back(PluginField("image", nullptr, PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(PluginField("featureMapShapesH", nullptr, PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(PluginField("featureMapShapesW", nullptr, PluginFieldType::kINT32, 1));
	mFC.nbFields = mPluginAttributes.size();
	mFC.fields = mPluginAttributes.data();
}

const char* GridAnchorBasePluginCreator::getPluginName() const
{
	return mPluginName;
}

const char* GridAnchorBasePluginCreator::getPluginVersion() const
{
	return GRID_ANCHOR_PLUGIN_VERSION;
}

const PluginFieldCollection* GridAnchorBasePluginCreator::getFieldNames()
{
	return &mFC;
}

IPluginV2Ext* GridAnchorBasePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
	int numLayers = 1;
	std::vector<float> aspectRatios;
	// std::vector<int> fMapShapes;
	int featureMapShapesW, featureMapShapesH;
	std::vector<float> layerVariances;
	int anchorScale = 2;
	int level = 3, scalePerOctave = 2;
	// vector<int> image;
	int imgW, imgH;
	const PluginField* fields = fc->fields;

	const bool isFMapRect = (GRID_ANCHOR_PLUGIN_NAMES[1] == name);
	for (int i = 0; i < fc->nbFields; ++i)
	{
		const char* attrName = fields[i].name;
		if (!strcmp(attrName, "level"))
		{
			ASSERT(fields[i].type == PluginFieldType::kINT32);
			level = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
		}
		else if (!strcmp(attrName, "anchorScale"))
		{
			ASSERT(fields[i].type == PluginFieldType::kINT32);
			anchorScale = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
		}
		else if (!strcmp(attrName, "scalesPerOctave"))
		{
			ASSERT(fields[i].type == PluginFieldType::kINT32);
			scalePerOctave = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
		}
		else if (!strcmp(attrName, "variance"))
		{
			ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
			int size = fields[i].length;
			layerVariances.reserve(size);
			const auto* lVar = static_cast<const float*>(fields[i].data);
			for (int j = 0; j < size; j++)
			{
				layerVariances.push_back(*lVar);
				lVar++;
			}
		}
		else if (!strcmp(attrName, "anchorRatio"))
		{
			ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
			int size = fields[i].length;
			aspectRatios.reserve(size);
			const auto* aR = static_cast<const float*>(fields[i].data);
			for (int j = 0; j < size; j++)
			{
				aspectRatios.push_back(*aR);
				aR++;
			}
		}
		else if (!strcmp(attrName, "image"))
		{
			ASSERT(fields[i].type == PluginFieldType::kINT32);
			int size = fields[i].length;
			// image.reserve(size);
			const auto* img = static_cast<const int*>(fields[i].data);
			for (int j = 0; j < size; j++)
			{
				// image.push_back(*img);
				if (j == 0)
				{
					imgH = *img;
				}
				else
				{
					imgW = *img;
				}

				img++;
			}
		}
		else if (!strcmp(attrName, "featureMapShapesH"))
		{
			ASSERT(fields[i].type == PluginFieldType::kINT32);
			featureMapShapesH = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
		}
		else if (!strcmp(attrName, "featureMapShapesW"))
		{
			ASSERT(fields[i].type == PluginFieldType::kINT32);
			featureMapShapesW = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
		}
	}

	std::vector<GridAnchorParametersTf> boxParams(numLayers);

	// One set of box parameters for one layer
	for (int i = 0; i < numLayers; i++)
	{
		boxParams[i] = { level, scalePerOctave, aspectRatios.data(),(int)aspectRatios.size(),imgH, imgW,
		featureMapShapesW, featureMapShapesH, {layerVariances[0], layerVariances[1], layerVariances[2], layerVariances[3]} };
	}
	//    cout<<"new grid--------------- "<<endl;
	GridAnchorGenerator* obj = new GridAnchorGenerator(boxParams.data(), anchorScale, numLayers, mPluginName);
	obj->setPluginNamespace(mNamespace.c_str());
	return obj;
}

IPluginV2Ext* GridAnchorBasePluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
	// This object will be deleted when the network is destroyed, which will
	// call GridAnchor::destroy()
	//cout << "deserializePlugin" << endl;
	GridAnchorGenerator* obj = new GridAnchorGenerator(serialData, serialLength, mPluginName);
	obj->setPluginNamespace(mNamespace.c_str());
	return obj;
}

GridAnchorPluginCreator::GridAnchorPluginCreator()
{
	mPluginName = GRID_ANCHOR_PLUGIN_NAMES[0];
}

GridAnchorRectPluginCreator::GridAnchorRectPluginCreator()
{
	mPluginName = GRID_ANCHOR_PLUGIN_NAMES[1];
}
