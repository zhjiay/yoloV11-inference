#include "ClsInference.h"

#include <NvOnnxParser.h>

ClsInference::ClsInference(int numCls)
{
	_numClass = numCls;
}

ClsInference::~ClsInference()
{
	this->release();
}

bool ClsInference::load(const std::string& modelPath)
{
	std::string enginePath = modelPath.substr(0, modelPath.find_first_of(".")) + ".engine";
	std::string onnxPath = modelPath.substr(0, modelPath.find_first_of(".")) + ".onnx";
	if (!std::filesystem::exists(enginePath))
	{
		auto data = serialOnnxToEngine(onnxPath);
		std::ofstream outf(enginePath, std::ios::out | std::ios::binary);
		if (outf)
		{
			outf.write(data.data(), data.size());
			outf.close();
		}
		else
		{
			std::cerr << "open file error! : " << enginePath << std::endl;
		}
	}

	std::vector<char> data = loadEngineData(enginePath);
	_runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(_nvloger));
	_engine = std::unique_ptr<nvinfer1::ICudaEngine>(_runtime->deserializeCudaEngine(data.data(), data.size()));
	_context = std::unique_ptr<nvinfer1::IExecutionContext>(_engine->createExecutionContext());

	int ioNum = _engine->getNbIOTensors();
	if (ioNum != 2)	throw "error number of IO tensors!";

	_inputName = _engine->getIOTensorName(0);
	_inputDims = _engine->getTensorShape(_inputName.c_str());
	_imgH = _inputDims.d[2];
	_imgW = _inputDims.d[3];
	int inLen = 1;
	for (int i = 0; i < _inputDims.nbDims; i++) inLen *= (int)_inputDims.d[i];
	_hInData.resize(inLen);
	cudaMalloc((void**)&_dInPtr, inLen * sizeof(float));

	_outputName = _engine->getIOTensorName(1);
	_outputDims = _engine->getTensorShape(_outputName.c_str());
	int outLen = 1;
	for (int i = 0; i < _outputDims.nbDims; i++) outLen *= (int)_outputDims.d[i];
	_hOutData.resize(outLen);
	cudaMalloc((void**)&_dOutPtr, outLen * sizeof(float));

	warmUp();
	return true;
}

bool ClsInference::inference(cv::Mat& inputMat, int& classId)
{
	preProcess(inputMat, _hInData);

	cudaError cErr = cudaSuccess;
	cErr = cudaMemcpy(_dInPtr, _hInData.data(), _hInData.size() * sizeof(float), cudaMemcpyHostToDevice);
	if (cErr!=cudaSuccess)
	{
		std::cerr << "cudaMemcpy input error\n";
		return false;
	}

	std::vector<void*> bindings = { _dInPtr, _dOutPtr };
	bool flag = _context->executeV2(bindings.data());
	if (!flag)
	{
		std::cerr << "context execute error\n";
		return false;
	}

	cErr = cudaMemcpy(_hOutData.data(), _dOutPtr, _hOutData.size() * sizeof(float), cudaMemcpyDeviceToHost);
	if (cErr!=cudaSuccess)
	{
		std::cerr << "cudaMemcpy output error\n";
		return false;
	}

	classId = postProcess(_hOutData);
	return true;
}

void ClsInference::release()
{
	cudaFree(_dInPtr);
	cudaFree(_dOutPtr);
	_context.reset();
	_engine.reset();
}

void ClsInference::warmUp()
{
	cv::Mat mat(_imgH, _imgW, CV_8UC3);
	cv::randu(mat, cv::Scalar::all(0), cv::Scalar::all(255));
	int classId = -1;
	inference(mat, classId);
}

void ClsInference::preProcess(cv::Mat& inputMat, std::vector<float>& hData)
{
	assert(inputMat.type() == CV_8UC3);
	cv::Mat rMat;

	if (inputMat.rows != _imgH || inputMat.cols != _imgW)
	{
		double scale = std::max((1.0 * inputMat.cols) / _imgW, (1.0 * inputMat.rows) / _imgH);
		cv::Size newSz(inputMat.cols / scale, inputMat.rows / scale);
		if (newSz.width > _imgW) newSz.width = _imgW;
		if (newSz.height > _imgH) newSz.height = _imgH;
		cv::resize(inputMat, rMat, newSz);
		cv::copyMakeBorder(rMat, rMat, 0, _imgH - rMat.rows, 0, _imgW - rMat.cols, CV_HAL_BORDER_CONSTANT, cv::Scalar::all(0));
	}
	else
	{
		rMat = inputMat;
	}
	auto ptr = rMat.data;
	auto hPtr = hData.data();
	for (int r = 0; r < _imgH; r++)
	{
		for (int c = 0; c < _imgW; c++)
		{
			hPtr[2 * _imgH * _imgW + r * _imgW + c] = static_cast<float>(ptr[3 * (r * _imgW + c) + 0]) / 255.0f;
			hPtr[1 * _imgH * _imgW + r * _imgW + c] = static_cast<float>(ptr[3 * (r * _imgW + c) + 1]) / 255.0f;
			hPtr[0 * _imgH * _imgW + r * _imgW + c] = static_cast<float>(ptr[3 * (r * _imgW + c) + 2]) / 255.0f;
		}
	}
}

int ClsInference::postProcess(std::vector<float>& output)
{
	int maxIdx=std::max_element(output.begin(), output.end()) - output.begin();
	return maxIdx;
}

std::vector<char> ClsInference::loadEngineData(std::string enginePath){
	std::ifstream inf(enginePath, std::ios::in | std::ios::binary);
	std::vector<char> data;
	if (inf)
	{
		inf.seekg(0, inf.end);
		size_t len = inf.tellg();
		inf.seekg(0, inf.beg);
		data.resize(len);

		inf.read(data.data(), data.size());
		inf.close();
	}
	return data;
}

std::vector<char> ClsInference::serialOnnxToEngine(std::string onnxPath)
{
	std::ifstream inf(onnxPath, std::ios::in | std::ios::binary);
	if (!inf)
	{
		std::cerr << "open file error! : " << onnxPath << std::endl;
	}
	inf.seekg(0, inf.end);
	size_t onnxDataSize = inf.tellg();
	inf.seekg(0, inf.beg);
	std::vector<char> onnxData(onnxDataSize);
	inf.read(onnxData.data(), onnxData.size());
	inf.close();

	NVlogger nvlog;
	auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(nvlog));
	auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
	auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, nvlog));

	bool isParsed = parser->parse(onnxData.data(), onnxData.size());

	int inputNum = network->getNbInputs();
	std::cout << "input num: " << inputNum << std::endl;
	for (int i = 0; i < inputNum; i++)
	{
		const auto input = network->getInput(i);
		const auto inputName = input->getName();
		const auto inputDim = input->getDimensions();
		std::cout << "input" << i << ":\t" << inputName << " size:[" << inputDim.d[0] << " " << inputDim.d[1] << " " << inputDim.d[2] << " " << inputDim.d[3] << "] DataType:" << (int)input->getType() << std::endl;
	}

	int outputNum = network->getNbOutputs();
	std::cout << "output num: " << outputNum << std::endl;
	for (int i = 0; i < outputNum; i++)
	{
		const auto output = network->getOutput(i);
		const auto outputName = output->getName();
		const auto outputDim = output->getDimensions();
		std::cout << "output" << i << ":\t" << outputName << " [";
		int outNbDims = outputDim.nbDims;
		for (int j = 0; j < outNbDims; j++)
		{
			std::cout << outputDim.d[j] << " ";
		}
		std::cout << "] DataType:" << (int)output->getType() << std::endl;
	}
	auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
	config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 30);
	nvinfer1::IHostMemory* hostSerialData = builder->buildSerializedNetwork(*network, *config);
	std::vector<char> serialData = std::vector<char>(static_cast<char*>(hostSerialData->data()), static_cast<char*>(hostSerialData->data()) + hostSerialData->size());
	return serialData;
}
