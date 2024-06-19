# coding:utf-8
import warnings
from ultralytics import YOLO

warnings.filterwarnings('ignore')
if __name__ == '__main__':
    model = YOLO('runs/detect/train9/weights/best.pt')  # 0.766 3006818 parameters, 0 gradients, 8.1 GFLOPs

    model.val(data='/Users/hong/Desktop/Project/psycho/su_detect/class_detect_2/datasets/dataset/data.yaml',
              split='test',
              imgsz=640,
              batch=4,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='class-detect',
              iou=0.5
              )
