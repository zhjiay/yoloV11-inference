using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using OpenCvSharp.Internal;
using OpenCvSharp.XPhoto;

namespace InferSegCSharp
{

    public class IoNode
    {
        public IoNode()
        {

        }

        public IoNode(string tName, int[] tShape)
        {
            Name=tName;
            Shape = new long[tShape.Length];
            int dataLen = 1;
            for (int i = 0; i < tShape.Length; i++)
            {
                Shape[i] = tShape[i];
                dataLen*= tShape[i];
            }
            Data = new float[dataLen];
        }

        public string Name { get; set; }
        public long[] Shape { get; set; }
        public float[] Data { get; set;}


        public override string ToString()
        {
            string s = Name + " (";
            for (int i = 0; i < Shape.Length; i++)
            {
                s += Shape[i].ToString() + ",";
            }
            s+= ")";
            return s;
        }
    }

    public class SegResult
    {
        public int Label;
        public float Confidence;
        public string LabelName;
        public List<Point> Contour;
        public Rect BoxRect;


    }

    public class SharpInference:IDisposable
    {
        public SharpInference(List<string> tLabelNames)
        {
            LabelNames=tLabelNames;
        }

        public void Load(string onnxPath, int platform = 0)
        {
            sessionOptions = new SessionOptions()
            {
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                EnableCpuMemArena = true,
            };
            if (platform == 0)
            {
                sessionOptions.AppendExecutionProvider_CUDA(0);
            }else if (platform == 1)
            {
                sessionOptions.AppendExecutionProvider_Tensorrt(0);
            }
            else
            {
                sessionOptions.AppendExecutionProvider_CPU(1);
            }

            session = new InferenceSession(onnxPath, sessionOptions);
            runOptions = new RunOptions();

            Input0Node = new IoNode(session.InputNames[0], session.InputMetadata[session.InputNames[0]].Dimensions);
            Output0Node = new IoNode(session.OutputNames[0], session.OutputMetadata[session.OutputNames[0]].Dimensions);
            Output1Node = new IoNode(session.OutputNames[1], session.OutputMetadata[session.OutputNames[1]].Dimensions);
            ImgHeight = session.InputMetadata[session.InputNames[0]].Dimensions[2];
            ImgWidth = session.InputMetadata[session.InputNames[0]].Dimensions[3];
            
            Console.WriteLine("********** Model Info **********");
            Console.WriteLine(Input0Node);
            Console.WriteLine(Output0Node);
            Console.WriteLine(Output1Node);
            Console.WriteLine("Begin Warm Up ...");
            var t0 = DateTime.Now;
            WarmUp();
            Console.WriteLine($" Warm Up Success in {(DateTime.Now-t0).TotalMilliseconds} ms");
        }

        public List<SegResult> Inference(Mat mat)
        {
            DateTime t0= DateTime.Now;
            PreProcess(mat);
            DateTime t1=DateTime.Now;
            
            try
            {
                using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(input0Node.Data, input0Node.Shape);
                var inputDic = new Dictionary<string, OrtValue> { { Input0Node.Name, inputOrtValue } };
                using var outputValues = session.Run(runOptions, inputDic, session.OutputNames);
                var outputSpan0 = outputValues[0].GetTensorDataAsSpan<float>().ToArray();
                Array.Copy(outputSpan0, Output0Node.Data, output0Node.Data.Length);
                var outputSpan1 = outputValues[1].GetTensorDataAsSpan<float>().ToArray();
                Array.Copy(outputSpan1, Output1Node.Data, Output1Node.Data.Length);
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
            }

            DateTime t2=DateTime.Now;
            List<SegResult> results=PostProcess();
            DateTime t3=DateTime.Now;

            if (doPerformaceMonitor)
            {
                double preMs = (t1 - t0).TotalMicroseconds / 1000.0;
                double runMs = (t2 - t1).TotalMicroseconds / 1000.0;
                double postMs = (t3 - t2).TotalMicroseconds / 1000.0;
                performaceDataList.Add(new Tuple<double, double, double, int>(preMs, runMs, postMs, results.Count));
                Console.WriteLine($"{preMs.ToString("0.00")}\t{runMs.ToString("0.00")}\t{postMs.ToString("0.00")}\t{(results.Count==0?0:postMs/results.Count).ToString("0.00")}\t{results.Count}");
            }
            return results;
        }


        private void PreProcess(Mat inputMat)
        {
            Mat tMat = new Mat();
            if (inputMat.Rows != ImgHeight || inputMat.Cols != ImgWidth)
            {
                Cv2.Resize(inputMat, tMat, new Size(ImgWidth, ImgHeight));
            }
            else
            {
                tMat = inputMat;
            }
            Vec3b[] bgrBytes=new Vec3b[ImgWidth*ImgHeight];
            tMat.GetArray<Vec3b>(out bgrBytes);
            Span<float> inSpan = Input0Node.Data.AsSpan();
            ReadOnlySpan<Vec3b> matSpan = bgrBytes.AsSpan();

            // C# 访问内存使用Span接近C++的速度，这里使用原始数组和Mat访问数据，约为200ms，使用Span约为2ms
            for (int r = 0; r < ImgHeight; r++)
            {
                for (int c = 0; c < ImgWidth; c++)
                {
                    inSpan[2 * ImgHeight*ImgWidth + r * ImgWidth + c] = ((float)matSpan[(r * ImgWidth + c)][0]) / 255.0f;
                    inSpan[1 * ImgHeight * ImgWidth + r * ImgWidth + c] = ((float)matSpan[(r * ImgWidth + c)][1]) / 255.0f;
                    inSpan[0 * ImgHeight * ImgWidth + r * ImgWidth + c] = ((float)matSpan[ (r * ImgWidth + c)][2]) / 255.0f;
                }
            }
        }

        private List<SegResult> PostProcess()
        {
            List<SegResult> results=new List<SegResult>();

            int numClass = LabelNames.Count;
            List<int> shape0 = new List<int>();
            for (int i = 0; i < output0Node.Shape.Length; i++)
            {
                shape0.Add((int)output0Node.Shape[i]);
            }
            List<int> shape1 = new List<int>();
            for (int i = 0; i < output1Node.Shape.Length; i++)
            {
                shape1.Add((int)output1Node.Shape[i]);
            }
            int maskDataLen = shape0[1] - 4 - numClass;



            Mat maskProto = new Mat(maskDataLen, shape1[2] * shape1[3], MatType.CV_32FC1);
            maskProto.SetArray<float>(output1Node.Data);

            Mat output0Mat = new Mat(shape0[1], shape0[2], MatType.CV_32FC1);
            output0Mat.SetArray<float>(output0Node.Data);
            Mat output0T = output0Mat.T();

            List<Rect> bboxs = new List<Rect>();
            List<int> classIds = new();
            List<float> confidences = new();
            List<Mat> masks=new List<Mat>();

            for (int i = 0; i < output0T.Rows; i++)
            {
                Mat rowMat = output0T.Row(i);

                int maxClassId = 0;
                float maxConfidence = 0;
                for (int clsIdx = 0; clsIdx < numClass; clsIdx++)
                {
                    if (rowMat.At<float>(0, 4 + clsIdx) > maxConfidence)
                    {
                        maxConfidence = rowMat.At<float>(0, 4 + clsIdx);
                        maxClassId = clsIdx;
                    }
                }
                if(maxConfidence<CONFIDECE_THRESHOLD) continue;

                float x_center = rowMat.At<float>(0, 0);
                float y_center = rowMat.At<float>(0, 1);
                float width = rowMat.At<float>(0, 2);
                float height = rowMat.At<float>(0, 3);

                int x1 = (int)(Clamp(x_center - width / 2, 0, ImgWidth));
                int y1= (int)(Clamp(y_center - height / 2, 0,ImgHeight));
                int x2= (int)(Clamp(x_center + width / 2, 0,ImgWidth));
                int y2=(int)(Clamp(y_center + height / 2, 0,ImgHeight));
                bboxs.Add(new Rect(x1,y1, Int32.Max(x2-x1,0), Int32.Max(y2-y1, 0)));
                classIds.Add(maxClassId);
                confidences.Add(maxConfidence);

                Mat maskMat = new(1, shape0[1] - 4 - numClass, MatType.CV_32FC1);
                for (int j = 0; j < maskMat.Cols; j++)
                {
                    maskMat.At<float>(0, j) = rowMat.At<float>(0, j + 4 + numClass);
                }
                masks.Add(maskMat);
            }

            CvDnn.NMSBoxes(bboxs, confidences, NMS_SOCRE_THRESHOLD, NMS_IOU_THRESHOLD, out int[] indices);

            for (int i = 0; i < indices.Length; i++)
            {
                int idx=indices[i];

                Mat maskRes = (masks[idx]*maskProto).T();
                maskRes = maskRes.Reshape(1, shape1[2]);
                Cv2.Exp(-maskRes, maskRes);
                maskRes = 1.0 / (maskRes.Add(new Scalar(1.0)));
                Cv2.Resize(maskRes, maskRes, new Size(ImgWidth, ImgHeight), 0,0);

                Mat rectMask = Mat.Zeros(maskRes.Size(), maskRes.Type());
                rectMask.SubMat(bboxs[idx]).SetTo(new Scalar(1.0f));
                maskRes = maskRes.Mul(rectMask);

                Mat binaryMat = new Mat();
                Cv2.Threshold(maskRes, binaryMat, MASK_THRESHOLD, 1.0, ThresholdTypes.Binary);
                binaryMat.ConvertTo(binaryMat, MatType.CV_8UC1, 255);


                Point[][] contours;
                HierarchyIndex[] hierarchy;
                Cv2.FindContours(binaryMat,out contours,out hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

                List<Point> targetCnt = new();
                double maxSimilar = 0.0;
                for (int cntIdx = 0; cntIdx < contours.Length; cntIdx++)
                {
                    Rect bbRect = Cv2.BoundingRect(contours[cntIdx]);
                    double tSimilar=double.Min(bbRect.Width, bboxs[idx].Width)*double.Min(bbRect.Height, bboxs[idx].Height) / (bboxs[idx].Width * bboxs[idx].Height);
                    if (tSimilar > maxSimilar)
                    {
                        maxSimilar=tSimilar;
                        targetCnt = contours[cntIdx].ToList();
                    }
                }

                SegResult seg=new SegResult();
                seg.Label = classIds[idx];
                seg.LabelName = LabelNames[seg.Label];
                seg.Confidence = confidences[idx];
                seg.BoxRect = bboxs[idx];
                seg.Contour = targetCnt;
                results.Add(seg);
            }
            return results;
        }

        private void WarmUp()
        {
            Mat randMat = new Mat(ImgHeight, ImgWidth, MatType.CV_8UC3);
            Cv2.Randu(randMat, Scalar.All(0), Scalar.All(255));
            PreProcess(randMat);

            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(input0Node.Data, input0Node.Shape);
            var inputDic = new Dictionary<string, OrtValue> { { Input0Node.Name, inputOrtValue } };
            using var outputValues = session.Run(runOptions, inputDic, session.OutputNames);

            var outputSpan0 = outputValues[0].GetTensorDataAsSpan<float>().ToArray();
            Array.Copy(outputSpan0, Output0Node.Data, output0Node.Data.Length);
            
            var outputSpan1 = outputValues[1].GetTensorDataAsSpan<float>().ToArray();
            Array.Copy(outputSpan1, Output1Node.Data, Output1Node.Data.Length);
        }

        private bool doPerformaceMonitor = false;
        private List<Tuple<double, double, double, int>> performaceDataList = new List<Tuple<double, double, double, int>>();
        public void StartPerformaceMonitor()
        {
            doPerformaceMonitor = true;
        }

        public List<Tuple<double, double, double, int>> GetPerformaceData()
        {
            return performaceDataList;
        }


        private SessionOptions sessionOptions = null;
        private InferenceSession session = null;
        private RunOptions runOptions=null;
        
        private List<string> labelNames;
        public List<string> LabelNames
        {
            get { return labelNames; }
            private set { labelNames = value; }
        }

        private IoNode input0Node;
        public IoNode Input0Node
        {
            get { return input0Node; }
            private set { input0Node = value; }
        }

        private IoNode output0Node;
        public IoNode Output0Node
        {
            get { return output0Node; }
            private set { output0Node = value; }
        }

        private IoNode output1Node;

        public IoNode Output1Node
        {
            get { return output1Node; }
            private set { output1Node = value; }
        }

        private int imgWidth;

        public int ImgWidth
        {
            get { return imgWidth; }
            private set { imgWidth = value; }
        }

        private int imgHeight;

        public int ImgHeight
        {
            get { return imgHeight; }
            private set { imgHeight = value; }
        }

        public void Dispose()
        {
            
        }

        #region Const Params
        private const float CONFIDECE_THRESHOLD = 0.5f;
        private const float NMS_SOCRE_THRESHOLD = 0.45f;
        private const float NMS_IOU_THRESHOLD = 0.64f;
        private const float MASK_THRESHOLD = 0.3f;

        private float Clamp(float v, float min, float max)
        {
            return v > min ? (v < max ? v : max) : min;
        }

        #endregion
    }
}
