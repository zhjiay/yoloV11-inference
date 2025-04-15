#include "OnnxInference.h"

OnnxInference::OnnxInference(const std::vector<std::string>& labelNames)
{
	_labelNames = labelNames;
}

OnnxInference::~OnnxInference()
{
	this->release();
}

void OnnxInference::load(const std::string& onnxPath, int platform)
{
	std::filesystem::path  onnxP(onnxPath);
	if (!std::filesystem::exists(onnxP) || onnxP.extension()!=".onnx")
	{
		throw "onnxpath not exist error!";
	}

	const auto& available_providers = Ort::GetAvailableProviders();
	for (const auto& provider : available_providers) {
		std::cout << "Available provider: " << provider << std::endl;
	}

	
	Ort::SessionOptions session_options;
	if (platform==0)
	{
		// cuda mode
		OrtCUDAProviderOptions cuda_option;
		cuda_option.device_id = 0;
		cuda_option.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
		cuda_option.arena_extend_strategy = 0;
		cuda_option.arena_extend_strategy = 0;
		cuda_option.do_copy_in_default_stream = 1;
		session_options.AppendExecutionProvider_CUDA(cuda_option);
	}else if (platform==1)
	{
		OrtTensorRTProviderOptions tensorRT_option;
		tensorRT_option.device_id = 0;
		tensorRT_option.trt_max_workspace_size = 1 << 30;
		tensorRT_option.trt_fp16_enable = 0;//关闭fp16
		tensorRT_option.trt_int8_enable = 0;//关闭int8
		tensorRT_option.trt_engine_cache_enable = 1;
		std::string cachePath = (onnxP.parent_path() / onnxP.stem()).string() + "_cache";
		tensorRT_option.trt_engine_cache_path = cachePath.c_str();
		session_options.AppendExecutionProvider_TensorRT(tensorRT_option);
	}else
	{
		throw "error platform value!";
	}
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	session_options.SetIntraOpNumThreads(1);
	session_options.SetLogSeverityLevel(3);

	_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");
	std::wstring wOnnxPath(onnxPath.begin(), onnxPath.end());
	try
	{
		_sessionPtr = std::shared_ptr<Ort::Session>(new Ort::Session(_env, (ORTCHAR_T*)wOnnxPath.c_str(), session_options));
	}catch (const Ort::Exception& e)
	{
		std::cerr << e.what() << std::endl;
		throw "load error!";
	}

	_memoryInfo = std::make_shared<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));

	//获取输入输出信息, yoloV11输出是 1个input，2个output
	Ort::AllocatorWithDefaultOptions allocator;
	size_t inputCount = _sessionPtr->GetInputCount();
	if (inputCount != 1) throw "error input node count";
	Ort::AllocatedStringPtr inputNamePtr = _sessionPtr->GetInputNameAllocated(0, allocator);
	_input0Node.name = std::string(inputNamePtr.get());
	auto input0Info = _sessionPtr->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
	if (input0Info.GetElementType() != ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) throw"error input data type";
	_input0Node.shape = input0Info.GetShape();
	_input0Node.initData();//初始化 _inputTensor, 预先分配数据内存
	_inputHeight = _input0Node.shape[2];
	_inputWidth = _input0Node.shape[3];

	size_t outputCount = _sessionPtr->GetOutputCount();
	if (outputCount != 2) throw "error output node count";

	//获取output0
	Ort::AllocatedStringPtr output0NamePtr = _sessionPtr->GetOutputNameAllocated(0, allocator);
	_output0Node.name = std::string(output0NamePtr.get());
	auto output0Info = _sessionPtr->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
	if (output0Info.GetElementType() != ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) throw"error output0 data type";
	_output0Node.shape = output0Info.GetShape();
	_output0Node.initData();

	//获取output1
	Ort::AllocatedStringPtr output1NamePtr = _sessionPtr->GetOutputNameAllocated(1, allocator);	
	_output1Node.name = std::string(output1NamePtr.get());
	auto output1Info = _sessionPtr->GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo();
	auto output1Type = output1Info.GetElementType();
	if (output1Type != ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) throw"error output1 data type";
	_output1Node.shape = output1Info.GetShape();
	_output1Node.initData();

	std::cout << "********** load onnx **********\n";
	std::cout << _input0Node.toString() << std::endl;
	std::cout << _output0Node.toString() << std::endl;
	std::cout << _output1Node.toString() << std::endl;
	std::cout << "*******************************\n";

	std::cout << "Begin Warm Up... ";
	this->warnUp();
	std::cout << "Warm up success!\n";
}

void OnnxInference::inference(const cv::Mat& inputMat, std::vector<OnnxSegResult>& segResults)
{
	auto t0 = std::chrono::high_resolution_clock::now();
	preProcess(inputMat, _input0Node);
	auto t1 = std::chrono::high_resolution_clock::now();
	try
	{
		Ort::Value inputTensor = Ort::Value::CreateTensor<float>(*_memoryInfo, _input0Node.data.data(), _input0Node.data.size(), _input0Node.shape.data(), _input0Node.shape.size());
		std::vector<const char*> inputNames = { _input0Node.name.c_str() };
		std::vector<const char*> outputNames = {_output0Node.name.c_str(), _output1Node.name.c_str()};
		auto outputTensors=_sessionPtr->Run(_runOption, inputNames.data(),&inputTensor,1, outputNames.data(), outputNames.size());
		float* tempOutData0 = outputTensors[0].GetTensorMutableData<float>();
		memcpy(_output0Node.data.data(), tempOutData0, _output0Node.data.size() * sizeof(float));
		float* tempOutData1 = outputTensors[1].GetTensorMutableData<float>();
		memcpy(_output1Node.data.data(), tempOutData1, _output1Node.data.size() * sizeof(float));
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
	auto t2 = std::chrono::high_resolution_clock::now();
	postProcess(_output0Node, _output1Node, segResults);
	auto t3 = std::chrono::high_resolution_clock::now();
	if (_doPerformaceMonitor)
	{
		double preMs = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
		double runMs = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;
		double postMs = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / 1000.0;
		_performanceMonitor.emplace_back(preMs, runMs, postMs, segResults.size());

		std::cout << preMs << "\t" << runMs << "\t" << postMs << "\t" << (segResults.size() == 0 ? 0 : postMs / segResults.size()) << "\t" << segResults.size() << std::endl;
	}
}

void OnnxInference::release()
{
	_sessionPtr->release();
}

void OnnxInference::startPerformaceMonitor()
{
	_doPerformaceMonitor = true;
	_performanceMonitor.clear();
}

std::vector<std::tuple<double, double, double, int>> OnnxInference::getPerformaceData() const
{
	return _performanceMonitor;
}

void OnnxInference::preProcess(const cv::Mat& inputMat, IONode& inNode)
{
	cv::Mat cvtMat;
	if (inputMat.rows!=_inputHeight || inputMat.cols!= _inputWidth)
	{
		cv::resize(inputMat, cvtMat, cv::Size(_inputWidth, _inputHeight), 0, 0);
	}else
	{
		cvtMat = inputMat;
	}
	assert(inNode.data.size() == cvtMat.total() * cvtMat.channels());

	for (int r=0;r<cvtMat.rows;r++)
	{
		for (int  c=0;c<cvtMat.cols;c++)
		{
			inNode.data[2 * cvtMat.total() + r * cvtMat.cols + c] = static_cast<float>(cvtMat.at<cv::Vec3b>(r, c)[0])/255.0f;
			inNode.data[1 * cvtMat.total() + r * cvtMat.cols + c] = static_cast<float>(cvtMat.at<cv::Vec3b>(r, c)[1])/255.0f;
			inNode.data[0 * cvtMat.total() + r * cvtMat.cols + c] = static_cast<float>(cvtMat.at<cv::Vec3b>(r, c)[2])/255.0f;
		}
	}
}

void OnnxInference::postProcess(const IONode& outNode0, const IONode& outNode1, std::vector<OnnxSegResult>& segResults)
{
	const int numClass = _labelNames.size();
	auto shape0 = outNode0.shape;
	auto shape1 = outNode1.shape;
	auto data0 = outNode0.data;
	auto data1 = outNode1.data;
	const int maskDataLen = shape0[1] - 4 - numClass;
	assert(maskDataLen == 32);//一般是32位掩码

	cv::Mat mask_proto(maskDataLen, shape1[2] * shape1[3], CV_32FC1, data1.data());
	cv::Mat output0Mat(shape0[1], shape0[2], CV_32FC1, data0.data());
	output0Mat = output0Mat.t();

	std::vector<cv::Rect> bboxs;
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Mat> masks;

	for (int i=0; i<output0Mat.rows; i++)
	{
		float* rowPtr = output0Mat.ptr<float>(i);
		int classId = std::max_element(rowPtr + 4, rowPtr + 4 + numClass) - (rowPtr + 4);
		float confidence = rowPtr[4 + classId];

		if (confidence < _confidence_threshold) continue;
		
		float x_center = rowPtr[0];
		float y_center = rowPtr[1];
		float width = rowPtr[2];
		float height = rowPtr[3];

		int x1 = static_cast<int>(clamp(x_center - width * 0.5, 0, _inputWidth));
		int y1 = static_cast<int>(clamp(y_center - height * 0.5, 0, _inputHeight));
		int x2 = static_cast<int>(clamp(x_center + width * 0.5, 0, _inputWidth));
		int y2 = static_cast<int>(clamp(y_center + height * 0.5, 0, _inputHeight));
		bboxs.emplace_back(x1, y1, std::max(x2 - x1, 0), std::max(y2 - y1, 0));
		classIds.emplace_back(classId);
		confidences.emplace_back(confidence);
		cv::Mat maskMat(1, shape0[1] - 4 - numClass, CV_32FC1, (rowPtr + 4 + numClass));
		masks.emplace_back(maskMat);
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(bboxs, confidences, _nms_socre_threshold, _nms_iou_threshold, indices,0.5);

	segResults.clear();
	segResults.reserve(indices.size());
	for (int idx : indices)
	{
		cv::Mat mask = (masks[idx] * mask_proto).t();
		mask = mask.reshape(1, shape1[2]);
		cv::exp(-mask, mask);
		mask = 1.0 / (1.0 + mask);
		cv::resize(mask, mask, cv::Size(_inputWidth, _inputHeight), 0, 0);

		//使用bbox 提取目标区域的掩膜
		cv::Mat rectMask = cv::Mat::zeros(mask.size(), mask.type());
		rectMask(bboxs[idx]) = 1.0f;
		mask = mask.mul(rectMask);

		cv::Mat mask_binary;
		cv::threshold(mask, mask_binary, _mask_threshold, 1.0, cv::THRESH_BINARY);
		mask_binary.convertTo(mask_binary, CV_8UC1, 255);

		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(mask_binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		//可能会找到多个contour，其中contour的boundRect最接近 bbox的是目标结果
		std::vector<cv::Point> targetCnt;
		double maxSimilar = 0.0;
		for (int cntIdx=0; cntIdx<contours.size(); cntIdx++)
		{
			cv::Rect boundRect = cv::boundingRect(contours[cntIdx]);
			double tSimilar = std::min(boundRect.width, bboxs[idx].width) * std::min(boundRect.height, bboxs[idx].height) * 1.0 / bboxs[idx].area();
			if (tSimilar>maxSimilar)
			{
				maxSimilar = tSimilar;
				targetCnt = contours[cntIdx];
			}
		}

		OnnxSegResult seg;
		seg.label = classIds[idx];
		seg.labelName = _labelNames[seg.label];
		seg.confidence = confidences[idx];
		seg.box = bboxs[idx];
		seg.contour = targetCnt;
		segResults.emplace_back(seg);
	}
}

void OnnxInference::warnUp()
{
	cv::Mat mat(_inputHeight, _inputWidth, CV_32FC3);
	cv::randu(mat, cv::Scalar::all(0), cv::Scalar::all(1));

	memcpy(_input0Node.data.data(), mat.data, _input0Node.data.size() * sizeof(float));
	try
	{
		Ort::Value inputTensor = Ort::Value::CreateTensor<float>(*_memoryInfo, _input0Node.data.data(), _input0Node.data.size(), _input0Node.shape.data(), _input0Node.shape.size());
		std::vector<const char*> inputNames = { _input0Node.name.c_str() };
		std::vector<const char*> outputNames = { _output0Node.name.c_str(), _output1Node.name.c_str() };
		auto outputTensors = _sessionPtr->Run(_runOption, inputNames.data(), &inputTensor, 1, outputNames.data(), outputNames.size());
		float* tempOutData0 = outputTensors[0].GetTensorMutableData<float>();
		memcpy(_output0Node.data.data(), tempOutData0, _output0Node.data.size() * sizeof(float));
		float* tempOutData1 = outputTensors[1].GetTensorMutableData<float>();
		memcpy(_output1Node.data.data(), tempOutData1, _output1Node.data.size() * sizeof(float));

	}catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
}
