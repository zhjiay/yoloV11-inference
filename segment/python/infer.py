import csv
from torch.backends.mkl import verbose
from ultralytics import YOLO
import numpy as np
import cv2 as cv
import time
import os
import glob

def inferTestImage():
    modelPath=r"C:\aispace\seg\models\igbtmodels\new_N_17_800_40model\weights\best.pt"
    model=YOLO(modelPath, task="segment")
    imgPath=r"C:\aispace\seg\test\test0320\images\img.png"
    image=cv.imread(imgPath, cv.IMREAD_REDUCED_COLOR_2)
    rows=image.shape[0]
    cols=image.shape[1]
    imgSZ=800
    overlap=80
    xNum=int(rows/(imgSZ-overlap)+1)
    yNum=int(cols/(imgSZ-overlap)+1)
    xDelta=(rows-1.0-imgSZ)/(xNum-1)
    yDelta=(cols-1.0-imgSZ)/(yNum-1)

    ms=[]
    for x in range(xNum):
        for y in range(yNum):
            t0=time.time()
            xPos=int(x*xDelta)
            yPos=int(y*yDelta)
            roi=image[xPos:xPos+imgSZ,yPos:yPos+imgSZ]
            results=model(roi, imgsz=imgSZ, verbose=False)
            for result in results:
                # print(result.boxes)
                result.save(filename=f"C:\\aispace\\seg\\test\\test0320\\images\\result_py\\{x}_{y}.jpeg")
                # result.show()
            t1=time.time()
            ms.append(t1-t0)
    print(ms)

def InferTotalData():
    colors=[(0,0,255),(0,255,0),(255,0,0),(0,255,255)]
    model=YOLO(r"C:\aispace\seg\models\igbtmodels\new_N_17_800_40model\weights\best.pt", task="segment")
    imgDir=r"C:\aispace\seg\data\images"
    imgPaths=glob.glob(os.path.join(imgDir,"**","*.jpeg"), recursive=True)
    print(f"img count: {len(imgPaths)}")

    costMs=[]
    for i,imgPath in enumerate(imgPaths):
        print(i)
        img=cv.imread(imgPath, cv.IMREAD_COLOR)
        results=model(img, imgsz=800, verbose=False)
        boxes=results[0].boxes
        masks=results[0].masks
        probs=results[0].probs
        names=results[0].names
        speed=results[0].speed
        drawMat=img.copy()
        speed["count"]=0 if masks is None else len(masks)
        costMs.append(speed)
        print(speed)
        if masks is not None and boxes is not None:
            xy_coords=masks.xy
            count=len(boxes.cls)
            for i in range(count):
                classId=int(boxes.cls[i].item())
                confidence=boxes.conf[i].item()
                box=boxes.xywh[i].cpu().numpy().tolist()
                cnt=xy_coords[i].astype(int)
                if cnt is None or len(cnt)==0:
                    continue
                cv.drawContours(drawMat, [cnt], -1, colors[classId], thickness=-1)
        # resMat=cv.addWeighted(drawMat,0.5, img, 0.5, 0)
        # saveDir=r"C:\aispace\seg\data\results\result_yolo"
        # savePath=saveDir+"\\"+imgPath.split("\\")[-1]
        # cv.imwrite(savePath,resMat)
    csvDir=r"C:\aispace\seg\data\results\performace_pythonPT.csv"
    with open(csvDir, "w", newline="") as f:
        fieldNames=costMs[0].keys() if costMs else []
        writer=csv.DictWriter(f, fieldnames=fieldNames)
        writer.writeheader()
        writer.writerows(costMs)
        print("write csv over")

if __name__ == '__main__':
    InferTotalData()