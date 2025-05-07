# yolo det

## yolo det推理基本流程

### PreProcess
通用预处理，将bgrbgrbgr的值进行归一化并重新排列为rrrgggbbb的格式，并将值归一化到0-1

### PostProcess
yolo-det只有一个分割结果，output的Dims为[BatchSize,4+numClass, resultCount]。其中每一列为一组数据，前四位表示
x_center, y_center, width, height,均为基于原始图像大小的维度。之后的值为每个类别的概率分数，可以从中筛选出超过分数阈值的数据。与obb的区别在于少了最后一位弧度值。

### 模型数据
基于1280*1280图片，yoloV11-M obb模型进行tensorRT推理。GPU为4070tisuper
| model | params(M) | FLOPS(B) | runtime(MB) | engine(MB) | context(MB) | used GPU Mem (MB) | preProcess(ms) | Inference(ms) | PostProcess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m | 20.1 | 68.0 | - | - | - | - | - | - | - |