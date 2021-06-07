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
#ifndef TRT_KERNEL_H
#define TRT_KERNEL_H

#include "cublas_v2.h"
#include "plugin.h"
#include <cassert>
#include <cstdio>
#include "struct.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using namespace myPlugin;
#define DEBUG_ENABLE 0

typedef enum
{
    NCHW = 0,
    NC4HW = 1
} DLayout_t;

pluginStatus_t allClassNMS(cudaStream_t stream, int num, int num_classes, int num_preds_per_class, int top_k,
    float nms_threshold, bool share_location, bool isNormalized, DataType DT_SCORE, DataType DT_BBOX, void* bbox_data,
    void* beforeNMS_scores, void* beforeNMS_index_array, void* afterNMS_scores, void* afterNMS_index_array,
    bool flipXY = false);

extern "C" pluginStatus_t detectionInference(cudaStream_t stream, int N, int C1, int C2, bool shareLocation,
    bool varianceEncodedInTarget, int backgroundLabelId, int numPredsPerClass, int numClasses, int topK, int keepTopK,
    float confidenceThreshold, float nmsThreshold, CodeTypeSSD codeType, DataType DT_BBOX, const void* locData,
    const void* priorData, DataType DT_SCORE, const void* confData, void* keepCount, void* topDetections,
    void* workspace, bool isNormalized = true, bool confSigmoid = false

);

pluginStatus_t gatherTopDetections(cudaStream_t stream, bool shareLocation, int numImages, int numPredsPerClass,
    int numClasses, int topK, int keepTopK, DataType DT_BBOX, DataType DT_SCORE, const void* indices,
    const void* scores, const void* bboxData, void* keepCount, void* topDetections);

size_t detectionForwardBBoxDataSize(int N, int C1, DataType DT_BBOX);

size_t detectionForwardBBoxPermuteSize(bool shareLocation, int N, int C1, DataType DT_BBOX);

size_t sortScoresPerClassWorkspaceSize(int num, int num_classes, int num_preds_per_class, DataType DT_CONF);

size_t sortScoresPerImageWorkspaceSize(int num_images, int num_items_per_image, DataType DT_SCORE);

pluginStatus_t sortScoresPerImage(cudaStream_t stream, int num_images, int num_items_per_image, DataType DT_SCORE,
    void* unsorted_scores, void* unsorted_bbox_indices, void* sorted_scores, void* sorted_bbox_indices,
    void* workspace);

pluginStatus_t sortScoresPerClass(cudaStream_t stream, int num, int num_classes, int num_preds_per_class,
    int background_label_id, float confidence_threshold, DataType DT_SCORE, void* conf_scores_gpu,
    void* index_array_gpu, void* workspace);

size_t calculateTotalWorkspaceSize(size_t* workspaces, int count);

const char* cublasGetErrorString(cublasStatus_t error);

pluginStatus_t permuteData(cudaStream_t stream, int nthreads, int num_classes, int num_data, int num_dim,
    DataType DT_DATA, bool confSigmoid, const void* data, void* new_data);

size_t detectionForwardPreNMSSize(int N, int C2);

size_t detectionForwardPostNMSSize(int N, int numClasses, int topK);

extern "C" pluginStatus_t decodeBBoxes(cudaStream_t stream, int nthreads, CodeTypeSSD code_type, bool variance_encoded_in_target,
    int num_priors, bool share_location, int num_loc_classes, int background_label_id, bool clip_bbox, DataType DT_BBOX,
    const void* loc_data, const void* prior_data, void* bbox_data);

extern "C" pluginStatus_t anchorGridInference(cudaStream_t stream, GridAnchorParametersTf param, int numAspectRatios,
    const void* aspectRatios, const void* scales, void* outputData);

extern "C" pluginStatus_t anchorGridTorch(cudaStream_t stream, AnchorParamsTorch params, void *output);

extern "C" void* detectionInferenceTorch(cudaStream_t stream, int batchSize, void *workspace, const void *anchor, const void *confData, const void *locData, void *boxPtr,
	void *scorePtr, void *classPtr, int layerIndex, int featureSize, int priorNum, int classNum, int keepK, int layerNum, float scoreThreshold, int keepTopK);

extern "C" pluginStatus_t batchNms(cudaStream_t stream, int batchSize, void *workspace, const void *box, const void *score, const void *classIndex, float iouthreshold, int classNum, int layerNum, int topK
	, int srcW, int srcH, int tarW, int tarH, float* outLoc, float* outConf, int* outClass, int keepTopK);

size_t intSize(int N, int size);

size_t floatSize(int N, int size);

size_t sortedSize(int size);

size_t permuteDataSize(int N, int size);

size_t sigmoidDataSize(int N, int size);

size_t indexDataSize(int N, int size);

size_t predictDataSize(int N, int size);

size_t unsigned_long_long_size(int N, int size);

size_t boolDataSize(int N, int size);

extern "C" pluginStatus_t yolov3Anchor(cudaStream_t stream, AnchorParamsYolov3 params, void *output);
extern "C" pluginStatus_t yolov3Detection(cudaStream_t stream, int batchSize, void *workspace, const void* predMap, const void* anchor,
	void* boxOut, void* clsScoreOut, void* confScoreOut, int clsNum, int layerIndex, int featureSize, int stride, int concatIndex, int sum);
extern "C" pluginStatus_t yolov3Nms(cudaStream_t stream, int batchSize, void *workspace, const float* mlvl_bboxes, const float* mlvl_scores,
	const float* mlvl_conf_scores, float conf_thr, float score_thr, float boxScale0, float boxScale1, float boxScale2, float boxScale3, int sum, int clsNum, float* det_bboxes, int* det_labels, float* scores, float iouThreshold);

#endif
