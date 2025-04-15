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

output1的维度是[BatchSize,32, ImageHeight/4, ImageWidth/4]。从output0中提取的mask
