#ifndef ANCHOR_GENERATOR_TORCH_H
#define ANCHOR_GENERATOR_TORCH_H
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <cuda_runtime_api.h>
#include "NvInferRuntimeCommon.h"
#include <vector>
#include "plugin.h"
#include "kernel.h"
#include "NvInferPluginUtils.h"
#include "struct.h"

using namespace nvinfer1;
using namespace myPlugin;
namespace detectronPlugin
{

	class AnchorGeneratorTorch : public IPluginV2Ext
	{
	public:
		AnchorGeneratorTorch(const AnchorParamsTorch* paramsIn, int numLayer);

		AnchorGeneratorTorch(const void* data, size_t length);

		~AnchorGeneratorTorch() override = default;

		int getNbOutputs() const override;

		Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

		int initialize() override;

		void terminate() override;

		void destroy() override;

		size_t getWorkspaceSize(int) const override;

		int enqueue(
			int batch_size, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

		size_t getSerializationSize() const override;

		void serialize(void* buffer) const override;

		bool supportsFormat(DataType type, PluginFormat format) const override;

		const char* getPluginType() const override;

		const char* getPluginVersion() const override;

		IPluginV2Ext* clone() const override;

		void setPluginNamespace(const char* libNamespace) override;

		const char* getPluginNamespace() const override;

		DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

		bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

		bool canBroadcastInputAcrossBatch(int inputIndex) const override;


		void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
			const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
			const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) override;
	protected:
		const char* mPluginName;

	private:

	/*	std::vector<int> mSize;
		std::vector<int> mLayer;
		std::vector<int> mStride;*/
		int mLayerNum;
		//int mWidth, mHeigth;
		//int mOffset;
		const char* mPluginNamespace;
		std::string mNameSpace;
		//std::vector<int> mAspectRatio;
		std::vector<AnchorParamsTorch> mParams;
	};

	class GridAnchorBasePluginCreatorTorch : public BaseCreator
	{
	public:
		GridAnchorBasePluginCreatorTorch();

		~GridAnchorBasePluginCreatorTorch() override = default;

		const char* getPluginName() const override;

		const char* getPluginVersion() const override;

		const PluginFieldCollection* getFieldNames() override;

		IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) override;

		IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

	protected:
		const char* mPluginName;

	private:
		static PluginFieldCollection mFC;
		static std::vector<PluginField> mPluginAttributes;
	};

	//class GridAnchorPluginCreatorTorch : public GridAnchorBasePluginCreatorTorch
	//{
	//public:
	//	GridAnchorPluginCreatorTorch();
	//	~GridAnchorPluginCreatorTorch() override = default;
	//};
}

#endif // !ANCHOR_GENERATOR_TORCH_H
