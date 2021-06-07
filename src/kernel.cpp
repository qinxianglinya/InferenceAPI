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
#include "kernel.h"
#include "plugin.h"
#include <vector>
//#define NOMINMAX
#include <algorithm>
size_t detectionInferenceWorkspaceSize(bool shareLocation, int N, int C1, int C2, int numClasses, int numPredsPerClass,
    int topK, DataType DT_BBOX, DataType DT_SCORE)
{
    size_t wss[7];
    wss[0] = detectionForwardBBoxDataSize(N, C1, DT_BBOX);
    wss[1] = detectionForwardBBoxPermuteSize(shareLocation, N, C1, DT_BBOX);
    wss[2] = detectionForwardPreNMSSize(N, C2);
    wss[3] = detectionForwardPreNMSSize(N, C2);
    wss[4] = detectionForwardPostNMSSize(N, numClasses, topK);
    wss[5] = detectionForwardPostNMSSize(N, numClasses, topK);
    wss[6] = std::max(sortScoresPerClassWorkspaceSize(N, numClasses, numPredsPerClass, DT_SCORE),
        sortScoresPerImageWorkspaceSize(N, numClasses * topK, DT_SCORE));
    return calculateTotalWorkspaceSize(wss, 7);
}

size_t detectionTorchInferenceWorkspaceSize(int batchSize, int keepTopk, int nbLayer, int nbCls, DetectionOutputParametersTorch params, std::vector<int> featureSize, std::vector<int> topkCandidates)
{
	//size_t wss[15];
	//wss[0] = predictDataSize(batchSize, nbLayer * keepTopk * 4);
	//wss[1] = predictDataSize(batchSize, nbLayer * keepTopk * nbCls);
	//wss[2] = indexDataSize(batchSize, keepTopk * nbLayer);

	//wss[3] = permuteDataSize(batchSize, featureSize[0] * params.nbPriorbox * params.nbCls);
	//wss[4] = permuteDataSize(batchSize, featureSize[0] * params.nbPriorbox * 4);
	//wss[5] = intSize(batchSize, featureSize[0] * params.nbPriorbox * params.nbCls);
	//wss[6] = floatSize(batchSize, featureSize[0] * params.nbPriorbox * params.nbCls);
	//wss[7] = intSize(batchSize, featureSize[0] * params.nbPriorbox * params.nbCls);
	//wss[8] = floatSize(batchSize, topkCandidates[0]);
	//wss[9] = intSize(batchSize, topkCandidates[0]);
	//wss[10] = intSize(batchSize, topkCandidates[0]);
	//wss[11] = intSize(batchSize, topkCandidates[0]);
	//wss[12] = floatSize(batchSize, topkCandidates[0] * 4);
	//wss[13] = floatSize(batchSize, topkCandidates[0] * 4);
	//wss[14] = floatSize(batchSize, topkCandidates[0] * 4);


	//return calculateTotalWorkspaceSize(wss, 15);
	size_t wss[17];
	wss[0] = predictDataSize(batchSize, nbLayer * keepTopk * 4);
	wss[1] = predictDataSize(batchSize, nbLayer * keepTopk);
	wss[2] = indexDataSize(batchSize, keepTopk * nbLayer);

	wss[3] = floatSize(batchSize, featureSize[0] * params.nbPriorbox * params.nbCls);
	wss[4] = permuteDataSize(batchSize, featureSize[0] * params.nbPriorbox * 4);
	wss[5] = floatSize(batchSize, featureSize[0] * params.nbPriorbox * params.nbCls);
	wss[6] = intSize(batchSize, featureSize[0] * params.nbPriorbox * params.nbCls);
	wss[7] = (batchSize + 1) * sizeof(int);

	wss[8] = floatSize(batchSize, featureSize[0] * params.nbPriorbox * params.nbCls);

	wss[9] = intSize(batchSize, featureSize[0] * params.nbPriorbox * params.nbCls);

	wss[10] = floatSize(batchSize, topkCandidates[0]);

	wss[11] = intSize(batchSize, topkCandidates[0]);

	wss[12] = intSize(batchSize, topkCandidates[0]);

	wss[13] = intSize(batchSize, topkCandidates[0]);

	wss[14] = floatSize(batchSize, topkCandidates[0] * 4);
	wss[15] = floatSize(batchSize, topkCandidates[0] * 4);

	wss[16] = floatSize(batchSize, topkCandidates[0] * 4);

	return calculateTotalWorkspaceSize(wss, 17);

}

size_t yolov3NmsWorkspaceSize(int batchSize, Yolov3NmsParams params, int featureSize)
{
	size_t wss[23];
	wss[0] = floatSize(batchSize, featureSize * 3 * (4 + 1 + params.nbCls));
	wss[1] = floatSize(batchSize, featureSize * 3 * 2);
	wss[2] = floatSize(batchSize, featureSize * 3 * 2);
	wss[3] = floatSize(batchSize, featureSize * 3 * (5 + params.nbCls - 2));
	wss[4] = floatSize(batchSize, featureSize * 3 * 4);
	wss[5] = floatSize(batchSize, featureSize * 3);
	wss[6] = floatSize(batchSize, featureSize * 3);
	wss[7] = floatSize(batchSize, featureSize * 3);
	wss[8] = floatSize(batchSize, featureSize * 3);
	wss[9] = floatSize(batchSize, featureSize * 3);
	wss[10] = floatSize(batchSize, featureSize * 3);
	wss[11] = floatSize(batchSize, featureSize * 3);
	wss[12] = floatSize(batchSize, featureSize * 3);
	wss[13] = floatSize(batchSize, featureSize * 3 * 4);
	wss[14] = floatSize(batchSize, featureSize * 3);
	wss[15] = floatSize(batchSize, featureSize * 3 * (5 + params.nbCls - 4));
	wss[16] = floatSize(batchSize, featureSize * 3 * (5 + params.nbCls - 4));
	wss[17] = floatSize(batchSize, featureSize * 3);
	wss[18] = intSize(batchSize, featureSize * 3);
	wss[19] = intSize(batchSize, featureSize * 3);
	wss[20] = floatSize(batchSize, 1000 * 4);
	wss[21] = floatSize(batchSize, 1000 * params.nbCls);
	wss[22] = floatSize(batchSize, 1000);
	return calculateTotalWorkspaceSize(wss, 23);
}