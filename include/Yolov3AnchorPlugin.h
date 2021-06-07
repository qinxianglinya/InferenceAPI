#pragma once
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <cuda_runtime_api.h>
#include "NvInferRuntimeCommon.h"
#include <vector>
#include "plugin.h"
#include "kernel.h"
#include "NvInferPluginUtils.h"

using namespace nvinfer1;
namespace detectron2
{
	class Yolov3AnchorPlugin : public IPluginV2Ext
	{
	public:
		Yolov3AnchorPlugin(const AnchorParamsYolov3* paramsIn, int numLayer);

		Yolov3AnchorPlugin(const void* data, size_t length);

		~Yolov3AnchorPlugin() override = default;

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
		int mLayerNum;
		const char* mPluginNamespace;
		std::string mNameSpace;
		std::vector<AnchorParamsYolov3> mParams;
	};

	class Yolov3AnchorPluginCreatorTorch : public BaseCreator
	{
	public:
		Yolov3AnchorPluginCreatorTorch();

		~Yolov3AnchorPluginCreatorTorch() override = default;

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

}


