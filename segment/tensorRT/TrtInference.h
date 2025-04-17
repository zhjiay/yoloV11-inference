#pragma once

#include<iostream>
#include <vector>
#include<string>
#include<fstream>
#include <mutex>
#include<chrono>
#include <filesystem>
#include <random>
#include<opencv2/opencv.hpp>

#include <NvInfer.h>
#include<NvOnnxParser.h>
#include<cuda_runtime.h>
#include<cuda_runtime_api.h>

namespace inference_trt_space
{

	class NVlogger : public nvinfer1::ILogger
	{
	public:
		void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override
		{
			if (severity <= Severity::kINFO)
			{
				std::cout << msg << std::endl;
			}
		}
	};

	struct IONode
	{
		std::string name;
		nvinfer1::Dims dims;
		float* dPtr;
		float* hPtr;
		size_t dataLen;
		bool isMalloc = false;

		bool malloc()
		{
			dataLen = 1;
			for (int i = 0; i < dims.nbDims; i++)
			{
				dataLen *= dims.d[i];
			}
			hPtr = new float[dataLen];
			cudaError_t cudaError = cudaMalloc((void**)&dPtr, dataLen * sizeof(float));
			if (cudaError != cudaSuccess) return false;
			isMalloc = true;
			return isMalloc;
		}

		bool release()
		{
			if (isMalloc)
			{
				cudaFree(dPtr);
				delete hPtr;
				isMalloc = false;
			}
			return true;
		}

		std::string toString() const
		{
			std::string s = name + " :(";
			for (int i = 0; i < dims.nbDims; i++)
			{
				s += " " + std::to_string(dims.d[i]);
			}
			s += " )";
			return s;
		}
	};

	struct TrtSegResult
	{
		float confidence;
		int label;
		std::string labelName;
		std::vector<cv::Point> contour;
		cv::Rect box;
	};

	class TrtInference
	{
	public:
		TrtInference(const std::vector<std::string>& labelNames);
		~TrtInference();

		void load(const std::string& modelPath);
		void inference(const cv::Mat& inputMat, std::vector<TrtSegResult>& segResults);
		void release();

		void startPerformaceMonitor();
		std::vector<std::tuple<double, double, double, int>> getPerformaceData() const;

	private:
		void preProcess(const cv::Mat& inputMat, IONode& inNode);
		void postProcess(const IONode& outNode0, const IONode& outNode1, std::vector<TrtSegResult>& segResults);
		void postProcessMask(const IONode& outNode0, const IONode& outNode1, std::vector<TrtSegResult>& segResults);
		void analysisOnnxData(const std::string& modelPath, bool saveEngineData=true);

		void warnUp();

		inline static NVlogger _nvlogger;
		std::vector<char> _engineData;
		std::unique_ptr<nvinfer1::IRuntime> _runtime;
		std::unique_ptr<nvinfer1::ICudaEngine> _engine;
		std::unique_ptr<nvinfer1::IExecutionContext> _context;

		std::vector<std::string> _labelNames;

		int _inputWidth;
		int _inputHeight;

		IONode _input0Node;
		IONode _output0Node;
		IONode _output1Node;
		std::vector<void*> _ioBindings;

		std::atomic<bool> _isLoad;

		bool _doPerformaceMonitor = false;
		/// <summary>
		/// tuple=(preProcessTime, engineRunTime, postProcessTime, resultCount);
		/// </summary>
		std::vector<std::tuple<double, double, double, int>> _performanceMonitor;

#pragma region YOLOv11 后处理参数
		const float _confidence_threshold = 0.5f;
		const float _nms_socre_threshold = 0.45f;
		const float _nms_iou_threshold = 0.65f;
		const float _mask_threshold = 0.3f;

		float clamp(float val, float minVal, float maxVal)
		{
			return val > minVal ? (val < maxVal ? val : maxVal) : minVal;
		}
#pragma endregion

	};

}

