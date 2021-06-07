//#include <iostream>
#include "Engine.h"
#include <typeinfo>
#include<Windows.h>
#include <vector>
#include "NvInferPlugin.h"
#include "struct.h"
#include <io.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "trtInference.h"
#include <string.h>
#include <fstream>

using namespace engine;
using namespace std;
void loadImgFromFiles(const std::string &imgPath, const std::string &imgFormat, std::vector<std::string> &imgs)
{
	intptr_t hFile = 0;
	struct _finddata_t fileInfo;
	std::string str, fileFormatName;
	if (0 != strcmp(imgFormat.c_str(), ""))
	{
		fileFormatName = "\\*." + imgFormat;
	}
	else
	{
		fileFormatName = "\\*";
	}
	if ((hFile = _findfirst(str.assign(imgPath).append(fileFormatName).c_str(), &fileInfo)) != -1)
	{
		do
		{
			imgs.push_back(str.assign(imgPath).append("\\").append(fileInfo.name));
		} while (_findnext(hFile, &fileInfo) == 0);
		_findclose(hFile);
	}
}

int main()
{

	////测试分配显存对性能的影响
	//for (int i = 0; i < 2; i++)
	//{
	//	cudaSetDevice(i);
	//	uint8_t* test;
	//	cudaMalloc((void**)&test, 8000 * 26000 * 3 * 3 * sizeof(uint8_t));
	//	std::cout << "为显卡" << i << "分配显存" << std::endl;
	//}
	//step2
	//const char* modelName = "D:/trt engine 0119/tf-retinanet.trt";

	std::vector<std::string> defectName = { "gh", "yh", "ghW" };

	const char* modelName = "C:/Users/A1007w/Desktop/0525_test/d2-retinanet.trt";




	//prepare test image
	std::vector<std::string> imgName;
	const std::string imgPath = "C:/Users/A1007w/Desktop/0525_test/data1/";
	//const std::string imgPath = "D:/project1-2-5/data/";
	loadImgFromFiles(imgPath, "bmp", imgName);

	std::vector<cv::Mat> imgMat;

	imgData::_SHARE_IMAGE_DATA *img = new imgData::_SHARE_IMAGE_DATA[imgName.size()];
	std::cout << "start read img...." << std::endl;
	for (int i = 0; i < imgName.size(); i++)
	{
		imgMat.push_back(cv::imread(imgName[i]));
		(img[i]).nChannel = imgMat[i].channels();
		(img[i]).nWidth = imgMat[i].cols;
		(img[i]).nHeight = imgMat[i].rows;
		(img[i]).pData = imgMat[i].data;
		if (i == 0)
		{
			for (int j = 0; j < 50; j++)
			{
				std::cout << "img:" << (img[i]).pData[j] << std::endl;
			}
		}
	}
	std::cout << "end read img...." << std::endl;
	imgData::iDEFECT **result;
	result = new imgData::iDEFECT*[1024];
	for (int i = 0; i < 1024; i++)
	{
		result[i] = new imgData::iDEFECT[1024];
	}

	std::cout << "start initEngines...." << std::endl;
	TrtInference::initEngines(modelName);
	std::cout << "end initEngines...." << std::endl;

	//step3  
	std::cout << "start doInference" << std::endl;
	clock_t start = clock();

	for (int i = 0; i < 17000; i++)
	{
		std::cout << "i:" << i << std::endl;
		TrtInference::doInference(img, imgName.size(), result);
	}
	clock_t end = clock();
	std::cout << "time:" << double((end - start)) / CLOCKS_PER_SEC << "s" << std::endl;
	TrtInference::releaseEngine();
	std::cout << "----------" << std::endl;
	std::cout << "end doInference" << std::endl;
	std::cout << "end" << std::endl;


	////painting boxes, just for test
	std::string resultPath;
	resultPath = "C:/Users/A1007w/Desktop/0525_test/result/";
	/*
	for(int i=0; i<400; )*/
	for (int i = 0; i < imgName.size(); i++)
	{
		int k = 0;
		for (int j = 0; j < 1024; j++)
		{

			if (result[i][j].nScore > 0)
			{
				k++;
				if (i == 0)
				{
					std::cout << "score:" << result[i][j].nScore << " " << result[i][j].rc[0] << " " << result[i][j].rc[1] << " " << result[i][j].rc[2] << " " << result[i][j].rc[3] << std::endl;
				}	
				cv::rectangle(imgMat[i], cv::Point(result[i][j].rc[0], result[i][j].rc[1]), cv::Point(result[i][j].rc[2], result[i][j].rc[3]), cv::Scalar(255, 0, 0), 1, 8, 0);
				cv::putText(imgMat[i], defectName[result[i][j].nType] +":"+ to_string(result[i][j].nScore), cv::Point(result[i][j].rc[0], result[i][j].rc[1]), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(255, 0, 225));
				//cv::putText(imgMat[i], to_string(result[i][j].nType), cv::Point(result[i][j].rc[0], result[i][j].rc[1]), cv::FONT_HERSHEY_COMPLEX, 0.75, cv::Scalar(255, 0, 225));
			}
		}
		std::cout << "k:" << k << std::endl;
		int pos = imgName[i].find_last_of('\\');
		std::string name(imgName[i].substr(pos + 1));
		std::string picPath = resultPath + name;
		cv::imwrite(picPath, imgMat[i]);
	}
	std::cout << "infer end!!!!!!!!!" << std::endl;
	//release memory
	delete[]img;

	for (int i = 0; i < 1024; i++)
	{
		//std::cout << i << std::endl;
		if (result[i] != nullptr)
		{
			delete[] result[i];
		}
	}
	delete[]result;


	std::cout << "memory released" << std::endl;
	system("pause");
	return 1;
}
