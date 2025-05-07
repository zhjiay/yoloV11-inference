# yolo obb

## yolo obb推理基本流程

### PreProcess
通用预处理，将bgrbgrbgr的值进行归一化并重新排列为rrrgggbbb的格式，并将值归一化到0-1

### PostProcess
yolo-obb只有一个分割结果，output的Dims为[BatchSize,4+numClass+1, resultCount]。其中每一列为一组数据，前四位表示
x_center, y_center, width, height,均为基于原始图像大小的维度。之后的值为每个类别的概率分数，可以从中筛选出超过分数阈值的数据，最后以为是旋转框的弧度值。然后进行nms筛选。则可得最终结果。

### 模型数据
基于1280*1280图片，yoloV11-M obb模型进行tensorRT推理。GPU为4070tisuper
| model | params(M) | FLOPS(B) | runtime(MB) | engine(MB) | context(MB) | used GPU Mem (MB) | preProcess(ms) | Inference(ms) | PostProcess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m | 20.9 | 183.5 | 207.16 | 100.00 | 294.00 | 394.00 | 4.22 | 17.36 | - |


### 训练过程和数据集格式
``` python
    modelName="TestOBB_Model"
    model = YOLO("___Path_To_/yolo11m-obb.pt")  # load a pretrained model (recommended for training)
    results = model.train(data="dataset.yaml", epochs=200, imgsz=1280, batch=2, device="cuda:0", name=modelName) 
    model.export(format="onnx")
```
数据目录:  
images:  
---- |train  
---- |val  
labels:  
---- |train  
---- |val  

dataset.yaml:  
``` xml
train:  pathto/images/train   # 训练集图像路径
val: pathto/images/val   # 验证集图像路径
nc: 4                      # 类别数量
names: ['label0', 'label1', 'label2', 'label3']  # 类别名称
```
