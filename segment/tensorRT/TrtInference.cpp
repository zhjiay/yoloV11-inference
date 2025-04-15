#include "TrtInference.h"
namespace fs = std::filesystem;

inference_trt_space::TrtInference::TrtInference(const std::vector<std::string>& labelNames)
{
	_labelNames = labelNames;
	_isLoad = false;
}

inference_trt_space::TrtInference::~TrtInference()
{
	this->release();
}

void inference_trt_space::TrtInference::load(const std::string& modelPath)
{
	if (_isLoad) return;
	if (!fs::exists(modelPath)) throw "model not exists!";

	fs::path modelP(modelPath);
	auto exten = modelP.extension();
	if (exten==".onnx")
	{
		auto engineP = (modelP.parent_path() / modelP.stem()).string() + ".engine";
		if (fs::exists(engineP)) // onnx对应的engine文件存在，则直接读取engine文件
		{
			modelP = engineP;
		}else
		{
			analysisOnnxData(modelPath, true);
		}
	}else if (exten==".engine")
	{
		std::ifstream infile(modelP, std::ios::in | std::ios::binary);
		if (infile)
		{
			infile.seekg(0, infile.end);
			size_t dataLen = infile.tellg();
			infile.seekg(0, infile.beg);
			_engineData.resize(dataLen);
			infile.read(_engineData.data(), _engineData.size());
			infile.close();
		}else
		{
			throw "read engine data error!";
		}
	}//获取 engine data;
	
	_runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(_nvlogger));
	_engine = std::unique_ptr<nvinfer1::ICudaEngine>(_runtime->deserializeCudaEngine(_engineData.data(), _engineData.size()));
	if (_engine == nullptr) throw "engine error";
	_context = std::unique_ptr<nvinfer1::IExecutionContext>(_engine->createExecutionContext());
	if (_context == nullptr) throw "context error";

	int ioNum = _engine->getNbIOTensors();
	if (ioNum != 3) throw "io number error";

	_input0Node.name = _engine->getIOTensorName(0);
	_input0Node.dims = _engine->getTensorShape(_input0Node.name.c_str());
	if (!_input0Node.malloc()) throw "malloc input0 error";
	_inputHeight = _input0Node.dims.d[2];
	_inputWidth = _input0Node.dims.d[3];

	_output0Node.name = _engine->getIOTensorName(1);
	_output0Node.dims = _engine->getTensorShape(_output0Node.name.c_str());
	if (!_output0Node.malloc()) throw "malloc output0 error";

	_output1Node.name = _engine->getIOTensorName(2);
	_output1Node.dims = _engine->getTensorShape(_output1Node.name.c_str());
	if (!_output1Node.malloc()) throw "mallco output1 error";

	_ioBindings = { _input0Node.dPtr, _output0Node.dPtr, _output1Node.dPtr };	//设置 ioBindings;

	std::cout << "********** load tensorRT **********\n";
	std::cout << _input0Node.toString() << std::endl;
	std::cout << _output0Node.toString() << std::endl;
	std::cout << _output1Node.toString() << std::endl;
	std::cout << "*******************************\n";
	_isLoad = true;

	std::cout << "begin warm up...";
	this->warnUp();
	std::cout << "warm up over\n";
}

void inference_trt_space::TrtInference::inference(const cv::Mat& inputMat, std::vector<TrtSegResult>& segResults)
{

	auto t0 = std::chrono::high_resolution_clock::now();
	preProcess(inputMat, _input0Node);
	auto t1 = std::chrono::high_resolution_clock::now();

	cudaError_t cudaError;
	cudaError = cudaMemcpy(_input0Node.dPtr, _input0Node.hPtr, _input0Node.dataLen * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaError != cudaSuccess) throw "cudaMemcpy input0 error!";
	bool res = _context->executeV2(_ioBindings.data());
	if (!res) throw "inference error";
	cudaError = cudaMemcpy(_output0Node.hPtr, _output0Node.dPtr, _output0Node.dataLen * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaError != cudaSuccess) throw "cudaMemcpy output0 error!";
	cudaError = cudaMemcpy(_output1Node.hPtr, _output1Node.dPtr, _output1Node.dataLen * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaError != cudaSuccess) throw "cudaMemcpy output1.error1";

	auto t2 = std::chrono::high_resolution_clock::now();
	postProcess(_output0Node, _output1Node, segResults);
	auto t3 = std::chrono::high_resolution_clock::now();
	if (_doPerformaceMonitor)
	{
		double preMs = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
		double runMs = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;
		double postMs = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / 1000.0;
		_performanceMonitor.emplace_back(preMs, runMs, postMs, segResults.size());
		
		std::cout << preMs << "\t" << runMs << "\t" << postMs << "\t" << (segResults.size()==0?0:postMs /segResults.size()) << "\t" << segResults.size() << std::endl;
	}
}

void inference_trt_space::TrtInference::release()
{
	if (_isLoad)
	{
		_input0Node.release();
		_output0Node.release();
		_output1Node.release();
		_context.reset();
		_engine.reset();
		_runtime.reset();

		_isLoad = false;
	}
}

void inference_trt_space::TrtInference::startPerformaceMonitor()
{
	_doPerformaceMonitor = true;
	_performanceMonitor.clear();
}

std::vector<std::tuple<double, double, double, int>> inference_trt_space::TrtInference::getPerformaceData() const
{
	return _performanceMonitor;
}

void inference_trt_space::TrtInference::preProcess(const cv::Mat& inputMat, IONode& inNode)
{
	cv::Mat cvtMat;
	if (inputMat.rows!= _inputHeight || inputMat.cols != _inputWidth)
	{
		cv::resize(inputMat, cvtMat, cv::Size(_inputWidth, _inputHeight), 0, 0);
	}else
	{
		cvtMat = inputMat;
	}
	assert(inNode.dataLen == cvtMat.total() * cvtMat.channels());
	for (int r = 0; r < cvtMat.rows; r++)
	{
		for (int c = 0; c < cvtMat.cols; c++)
		{
			inNode.hPtr[2 * cvtMat.total() + r * cvtMat.cols + c] = static_cast<float>(cvtMat.at<cv::Vec3b>(r, c)[0]) / 255.0f;
			inNode.hPtr[1 * cvtMat.total() + r * cvtMat.cols + c] = static_cast<float>(cvtMat.at<cv::Vec3b>(r, c)[1]) / 255.0f;
			inNode.hPtr[0 * cvtMat.total() + r * cvtMat.cols + c] = static_cast<float>(cvtMat.at<cv::Vec3b>(r, c)[2]) / 255.0f;
		}
	}
}

void inference_trt_space::TrtInference::postProcess(const IONode& outNode0, const IONode& outNode1, std::vector<TrtSegResult>& segResults)
{
	auto shape0 = outNode0.dims; //[batch, boxsSize+numClass+32(mask number), totalResultCount]
	auto shape1 = outNode1.dims; //[batch, mask_number, _inputHeight/4, _inputWidht/4];
	const int numClass = _labelNames.size();
	const int maskDataLen = shape0.d[1] - 4 - numClass;
	assert(maskDataLen == 32);

	cv::Mat maskProto(maskDataLen, shape1.d[2] * shape1.d[3], CV_32FC1, outNode1.hPtr); // output1结果设置到 Mat上
	cv::Mat output0Mat(shape0.d[1], shape0.d[2], CV_32FC1, outNode0.hPtr); //output0 结果设置到 Mat上
	output0Mat = output0Mat.t();

	std::vector<cv::Rect> bboxs;
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Mat> masks;

	for (int i=0;i<output0Mat.rows;i++)
	{
		float* rowPtr = output0Mat.ptr<float>(i);
		for (int j = 0; j < 40; j++)
		{
			//std::cout << rowPtr[j] << " ";
		}
		int classId = std::max_element(rowPtr + 4, rowPtr + 4 + numClass) - (rowPtr + 4);
		float confidence = rowPtr[4 + classId];
		//std::cout << "(" << classId << " " << confidence << ")\n";
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
		cv::Mat maskMat(1, maskDataLen, CV_32FC1, (rowPtr + 4 + numClass));
		masks.emplace_back(maskMat);
	}
	std::vector<int> indices;
	cv::dnn::NMSBoxes(bboxs, confidences, _nms_socre_threshold, _nms_iou_threshold, indices, 0.5);

	segResults.clear();
	segResults.reserve(indices.size());
	for (int idx : indices)
	{
		cv::Mat mask = (masks[idx] * maskProto).t();
		mask = mask.reshape(1, shape1.d[2]);
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
		for (int cntIdx = 0; cntIdx < contours.size(); cntIdx++)
		{
			cv::Rect boundRect = cv::boundingRect(contours[cntIdx]);
			double tSimilar = std::min(boundRect.width, bboxs[idx].width) * std::min(boundRect.height, bboxs[idx].height) * 1.0 / bboxs[idx].area();
			if (tSimilar > maxSimilar)
			{
				maxSimilar = tSimilar;
				targetCnt = contours[cntIdx];
			}
		}

		TrtSegResult seg;
		seg.label = classIds[idx];
		seg.labelName = _labelNames[seg.label];
		seg.confidence = confidences[idx];
		seg.box = bboxs[idx];
		seg.contour = targetCnt;
		segResults.emplace_back(seg);
	}
}

void inference_trt_space::TrtInference::analysisOnnxData(const std::string& modelPath, bool saveEngineData)
{
	std::vector<char> onnxData;
	std::ifstream infile(modelPath, std::ios::in | std::ios::binary);
	if (infile)
	{
		infile.seekg(0, infile.end);
		size_t onnxDataLen = infile.tellg();
		infile.seekg(0, infile.beg);
		onnxData.resize(onnxDataLen);
		infile.read(onnxData.data(), onnxData.size());
		infile.close();
	}else
	{
		throw "read onnx data error!";
	}

	auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(_nvlogger));
	auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
	auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, _nvlogger));

	bool isParsed = parser->parse(onnxData.data(), onnxData.size());
	if (!isParsed) throw "parse onnx data error!";

	auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
	nvinfer1::IHostMemory* hostSerialData = builder->buildSerializedNetwork(*network, *config);
	_engineData = std::vector<char>(static_cast<char*>(hostSerialData->data()), static_cast<char*>(hostSerialData->data())+hostSerialData->size());

	if (saveEngineData)
	{
		fs::path modelP(modelPath);
		std::string enginePath = (modelP.parent_path() / modelP.stem()).string() + ".engine";
		std::ofstream outfile(enginePath, std::ios::out | std::ios::binary);
		if (outfile)
		{
			outfile.write(_engineData.data(), _engineData.size());
			outfile.clear();
		}
		else
		{
			throw "save engine data error!";
		}
	}
}

void inference_trt_space::TrtInference::warnUp()
{
	cv::Mat mat(_inputHeight, _inputWidth, CV_32FC3);
	cv::randu(mat, cv::Scalar::all(0), cv::Scalar::all(1));
	memcpy(_input0Node.hPtr, mat.ptr<float>(), _input0Node.dataLen * sizeof(float));
	cudaError_t cudaError;
	cudaError = cudaMemcpy(_input0Node.dPtr, _input0Node.hPtr, _input0Node.dataLen * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaError != cudaSuccess) throw "cudaMemcpy input0 error!";
	bool res = _context->executeV2(_ioBindings.data());
	if (!res) throw "inference error";
	cudaError = cudaMemcpy(_output0Node.hPtr, _output0Node.dPtr, _output0Node.dataLen * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaError != cudaSuccess) throw "cudaMemcpy output0 error!";
	cudaError = cudaMemcpy(_output1Node.hPtr, _output1Node.dPtr, _output1Node.dataLen * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaError != cudaSuccess) throw "cudaMemcpy output1.error1";
}
