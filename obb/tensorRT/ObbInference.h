#pragma once
#pragma warning(disable: 4996)
#define _CRT_SECURE_NO_WARNINGS

#include<iostream>
#include <vector>
#include<string>
#include<fstream>
#include <mutex>
#include<chrono>
#include <filesystem>
#include <omp.h>
#include<immintrin.h>
#include <random>
#include<opencv2/opencv.hpp>

#include<clipper2/clipper.h>

#include <NvInfer.h>
#include<NvOnnxParser.h>
#include<cuda_runtime.h>
#include<cuda_runtime_api.h>
#include <nvml.h>
#include<onnxruntime_cxx_api.h>

class NVlogger :public nvinfer1::ILogger {
public:
	void log(Severity severity, const char* msg) noexcept override
	{
		if (severity <= Severity::kINFO)
		{
			std::cout << msg << std::endl;
		}
	}
};

struct ObbResult
{
	int classId;
	float prob;
	cv::RotatedRect rrect;
};

class ObbInference
{
public:
	ObbInference(int numCls);
	~ObbInference();

	bool load(std::string modelPath);
	bool inference(cv::Mat& inputMat, std::vector<ObbResult>& results);
	void release();

private:
	void warmUp();
	void preProcess(cv::Mat& inputMat, std::vector<float>& hData);
	std::vector<ObbResult> postProcess(std::vector<float>& output);

	inline float clamp(float minVal, float val, float maxVal)
	{
		return val > maxVal ? maxVal : (val < minVal ? minVal : val);
	}

	std::vector<char> loadEngineData(std::string enginePath);
	std::vector<char> serialOnnxToEngine(std::string onnxPath);

	inline static NVlogger _nvloger;
	std::unique_ptr<nvinfer1::IRuntime> _runtime;
	std::unique_ptr<nvinfer1::ICudaEngine> _engine;
	std::unique_ptr<nvinfer1::IExecutionContext> _context;

	int _numClass;
	bool _isLoad;
	int _imgW;
	int _imgH;

	//io data;
	std::string _inputName;
	nvinfer1::Dims _inputDims;
	float* _dInPtr;
	std::vector<float> hInData;

	std::string _outputName;
	nvinfer1::Dims _outputDims;
	float* _dOutPtr;
	std::vector<float> hOutData;

	//const params
	const float _score_threshold = 0.35f;
	const float _nms_threshold = 0.75f;
};

