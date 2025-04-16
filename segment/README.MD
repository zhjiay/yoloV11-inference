# yolo segment

## yolo-seg 推理的基本流程

### PreProcess
这一步是将原始的图像数据转换为模型的目标输入尺寸，将bgrbgrbgr的值进行归一化并重新排列为rrrgggbbb的格式。

### Run
将第一步的数据传输，然后对于不同的推理框架调用运行接口进行推理。

### PostProcess
yolo-segment的分割结果分为两部分，output0,output1。假设Dims为[BatchSize, 3, ImageHeight, ImageWidth]。  
则output0的维度为[BatchSize,n0, n1]。这里每一列表示一个结果。n0的大小为 4(bbox)+classNum+32(mask data)，即:  
- 0-3(bbox) [x_center,y_center,widht,height] 均为输入图像尺寸下的坐标，非归一化。
- 3-3+classNum  [class0Prob,class1Prob,...,classNProb]所有类别的概率值都会列出，从这里找出最大的概率为此项实例  
- 后32为 mask系数，与output1做矩阵乘法得到mask。  

output1的维度是[BatchSize,32, dim2, dim3](其中dim2=InputHeight/4, dim3=InputHeight/4)。将output1转换为 [32, dim2*dim3]的Mat(BatchSize张Mat)。使用output0得到的mask[1,32]与 output1Mask[32, dim2*dim3]进行矩阵乘法，得到ResultMask[1,dim2*dim3]。再将ResultMask reshape为[dim2,dim3]的Mat。进行Sigmoid变化。将Mat概率转换到[0-1]之间。然后resize到输入图像尺寸[ImageHeight,ImageWidth]。进行二极化，然后需要使用output0获取的bbox，筛选轮廓区域，只有在bbox内且最大的轮廓才是目标轮廓,如果不筛选会存在其它区域的干扰轮廓,至此,获取到最终推理结果.

### 模型数据
基于tensorRT 推理的模型数据指标
| model | params(M) | FLOPS(B) | runtime(MB) | engine(MB) | context(MB) | used GPU mem(MB) | inference time(ms)  | io copy(ms) | execute(ms) |  
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n | 2.9 | 10.4 | 207.16 | 20.00 | 32.00 | 52.00 | 4.70 | 1.98 | 2.73 |
| s | 10.1 | 35.5 | 208.16 | 80.94 | 62.00 | 142.94 | 5.87 | 1.98 | 3.89 |
| m | 22.4 | 123.3 | 214.43 | 109.00 | 114.00 | 223.00 | 10.32 | 1.98 | 8.34 | 
| l | 27.6 | 142.2 | 207.16	| 138.00 | 118.00 |	256.00 | 12.08 | 1.98 |	10.11 |
| x | 62.1 | 319 | 207.16 |	306.00 | 178.00 | 484.00 | 19.88 | 1.98 | 17.91 |

### 不同平台推理数据
| platform | PreProcess | Inference | PostProcess | single PostProcess|
| --- | --- | --- | --- |
|trt| - | - | - |
|onnx-cuda| - | - | - |
|onnx-trt| - | - | - |
|C#-cuda| - | - | - |
|C#-trt| - | - | - |
|python| - | - | - |
