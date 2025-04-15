#include <iostream>
#include "OnnxInference.h"

void testInference();

int main()
{
    testInference();
    //std::cout << "Hello World!\n";
}

void testInference()
{
    std::vector<std::string> labelNames = { "ant","bear","cat", "dog" };
    std::vector<cv::Scalar> colors = { cv::Scalar(0,0,255), cv::Scalar(0,255,0),cv::Scalar(255,0,0),cv::Scalar(0,255,255) };
    std::string modelPath = "C:\\aispace\\seg\\code\\InferSeg\\models\\bestm.onnx";
    OnnxInference onnxInf(labelNames);
    onnxInf.load(modelPath,1);

    std::string imgDir = "C:\\aispace\\seg\\data\\images";
    std::vector<cv::String> imgPaths;
    cv::glob(imgDir + "\\*.jpeg", imgPaths);
    std::cout << "img count: " << imgPaths.size() << std::endl;

    onnxInf.startPerformaceMonitor();
    std::string resultDir = "C:\\aispace\\seg\\data\\results\\result_onnxTrt";
    for (int i=0;i<imgPaths.size();i++)
    {
        std::cout << i << std::endl;
        cv::Mat mat = cv::imread(imgPaths[i], cv::IMREAD_COLOR);
        std::vector<OnnxSegResult> segResults;
        onnxInf.inference(mat, segResults);

        //cv::Mat drawMat = mat.clone();
        //for (int j=0;j<segResults.size();j++)
        //{
        //    cv::drawContours(drawMat, std::vector<std::vector<cv::Point>>{segResults[j].contour}, -1, colors[segResults[j].label], -1);
        //}
        //cv::addWeighted(mat, 0.5, drawMat, 0.5, 0, drawMat);
        //std::string savePath=resultDir+"\\"+std::filesystem::path(imgPaths[i]).filename().string();
        //cv::imwrite(savePath, drawMat);
    }

    auto perfData = onnxInf.getPerformaceData();
    std::string csvPath = "C:\\aispace\\seg\\data\\results\\performance_onnxTrt.csv";
    std::ofstream outf(csvPath, std::ios::out);
    if (outf)
    {
        outf << "pre,run,post,count" << std::endl;
        for (int i = 0; i < perfData.size(); i++)
        {
            std::string line = std::to_string(std::get<0>(perfData[i])) + "," +
                std::to_string(std::get<1>(perfData[i])) + "," +
                std::to_string(std::get<2>(perfData[i])) + "," +
                std::to_string(std::get<3>(perfData[i]));
            outf << line << std::endl;
        }
        outf.close();
    }
    onnxInf.release();
}
