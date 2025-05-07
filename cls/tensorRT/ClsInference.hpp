#pragma once
#include <NvInfer.h>
#include <string>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>

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

class ClsInference
{
public:
	ClsInference(int numCls);
	~ClsInference();

	bool load(const std::string& modelPath);
	bool inference(cv::Mat& inputMat, int& classId);
	void release();

private:
	void warmUp();
	void preProcess(cv::Mat& inputMat, std::vector<float>& hData);
	int postProcess(std::vector<float>& outRes);

	std::vector<char> loadEngineData(std::string enginePath);
	std::vector<char> serialOnnxToEngine(std::string onnxPath);

	int _numClass;
	inline static NVlogger _nvloger;
	std::unique_ptr<nvinfer1::IRuntime> _runtime;
	std::unique_ptr<nvinfer1::ICudaEngine> _engine;
	std::unique_ptr<nvinfer1::IExecutionContext> _context;

	std::string _inputName;
	nvinfer1::Dims _inputDims;
	float* _dInPtr;
	std::vector<float> _hInData;

	std::string _outputName;
	nvinfer1::Dims _outputDims;
	float* _dOutPtr;
	std::vector<float> _hOutData;

	int _imgH;
	int _imgW;
};

