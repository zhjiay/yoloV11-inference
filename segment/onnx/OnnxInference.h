#pragma once

#include<vector>
#include <array>
#include<string>
#include<ctime>
#include<chrono>
#include<algorithm>
#include<filesystem>
#include<fstream>
#include<opencv2/opencv.hpp>
#include<onnxruntime_cxx_api.h>
#include<onnxruntime_c_api.h>

//struct IONode
//{
//	std::string nodeName;
//	size_t typeSize = sizeof(float);
//	std::vector<size_t> shape;
//};

struct IONode
{
	std::string name;
	std::vector<int64_t> shape;
	std::vector<float> data;
	size_t dataLen;

	void initData()
	{
		dataLen = 1;
		for (int i=0;i<shape.size();i++)
		{
			dataLen *= shape[i];
		}
		data.resize(dataLen);
	}

	std::string toString() const
	{
		std::string s = name + " [ ";
		for (int i=0;i<shape.size();i++)
		{
			s += std::to_string(shape[i]) + " ";
		}
		s += "]";
		return s;
	}
};

struct OnnxSegResult
{
	float confidence;
	int label;
	std::string labelName;
	std::vector<cv::Point> contour;
	cv::Rect box;
};

class OnnxInference
{
public:
	OnnxInference(const std::vector<std::string>& labelNames);
	~OnnxInference();

	void load(const std::string& onnxPath, int platform = 0);// 0=cuda 1=tensorRT
	void inference(const cv::Mat& inputMat, std::vector<OnnxSegResult>& segResults);
	void release();

	void startPerformaceMonitor();
	std::vector<std::tuple<double, double, double, int>> getPerformaceData() const;

private:
	void preProcess(const cv::Mat& inputMat, IONode& inNode);
	void postProcess(const IONode& outNode0, const IONode& outNode1, std::vector<OnnxSegResult>& segResults);

	void warnUp();

	std::vector<std::string> _labelNames;

	Ort::Env _env;
	std::shared_ptr<Ort::Session> _sessionPtr = nullptr;
	std::shared_ptr<Ort::MemoryInfo> _memoryInfo;
	const Ort::RunOptions  _runOption{ nullptr };

	IONode _input0Node;
	IONode _output0Node;
	IONode _output1Node;

	int _inputWidth;
	int _inputHeight;

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

	/// <summary>
	/// tuple=(preProcessTime, engineRunTime, postProcessTime, resultCount);
	/// </summary>
	std::vector<std::tuple<double, double, double, int>> _performanceMonitor;
	bool _doPerformaceMonitor = false;
};

