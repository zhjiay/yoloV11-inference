# yolo cls

## yolo cls推理基本流程

### PreProcess

通用预处理，将bgrbgrbgr的值进行归一化并重新排列为rrrgggbbb的格式，并将值归一化到0-1

### PostProcess

yolo-cls的结果是一个长度为numclass的数组[prob_0, prob_1, prob_2, ... ]。概率最大的值的index即代表类别。

### 模型数据
基于224*224的三通道图片， 使用yoloV11-m cls模型进行tensorRT推理，GPU为4070tisuper
| model | params(M) | FLOPS(B) | runtime(MB) | engine(MB) | context(MB) | used GPU Mem (MB) | preProcess(ms) | Inference(ms) | PostProcess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m | 10.4 | 5.0 | 207.16 | 44.00 | 10.0| 54.00 | 0.13 | 1.62 | ~0.0 |