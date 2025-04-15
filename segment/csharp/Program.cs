using Microsoft.ML.OnnxRuntime;
//安装onnx包，必须安装指定的包，如果是cuda就只能安装.Gpu包，不能安装其它默认类型的包
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Internal;

namespace InferSegCSharp
{
    public class Program
    {
        static void Main(string[] args)
        {
            TestInfer();
            Console.WriteLine("Hello, World!");
        }

        static void TestInfer()
        {
            List<string> labelNames = new List<string>() { "ant", "bear", "cat", "dog" };
            List<Scalar> colors = new List<Scalar>()
                { new Scalar(0, 0, 255), new Scalar(0, 255, 0), new Scalar(255, 0, 0), new Scalar(0, 255, 255) };

            SharpInference csInf = new SharpInference(labelNames);
            string modelPath = @"C:\aispace\seg\code\InferSeg\models\bestm.onnx";
            csInf.Load(modelPath,1);
            csInf.StartPerformaceMonitor();

            List<string> imgPaths = new List<string>();
            string imgDir = @"C:\aispace\seg\data\images";
            string resultDir = @"C:\aispace\seg\data\results\result_csharp";
            DirectoryInfo imgDirInfo = new DirectoryInfo(imgDir);
            foreach (var file in imgDirInfo.GetFiles())
            {
                imgPaths.Add(file.FullName);
            }
            Console.WriteLine($"img count: {imgPaths.Count}");
            for (int i = 0; i < imgPaths.Count; i++)
            {
                var subs = imgPaths[i].Split("\\");
                string name = subs[subs.Length - 1].Split(".")[0];

                Console.WriteLine($"{i}: ");
                Mat mat = Cv2.ImRead(imgPaths[i], ImreadModes.Color);
                List<SegResult> results = csInf.Inference(mat);

                //Mat drawMat = mat.Clone();
                //for (int j = 0; j < results.Count; j++)
                //{
                //    var cnts = new List<List<Point>>() { results[j].Contour };
                //    Cv2.DrawContours(drawMat, cnts, -1, colors[results[j].Label],-1);
                //}
                //Cv2.AddWeighted(mat, 0.5, drawMat,0.5, 0, drawMat);
                //string savePath = resultDir + "\\" + name + ".jpeg";
                //Cv2.ImWrite(savePath, drawMat);
            }

            var perfData = csInf.GetPerformaceData();
            List<string> lines = new List<string>();
            perfData.ForEach(item => {lines.Add($"{item.Item1},{item.Item2},{item.Item3},{item.Item4}");});
            string csvPath = "C:\\aispace\\seg\\data\\results\\performance_csharpTRT.csv";
            File.WriteAllLines(csvPath,lines);
        }

    }
}
