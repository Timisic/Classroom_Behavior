# coding:utf-8
from ultralytics import YOLO
import logging
import sys

# 设置日志记录
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# for handler in logging.root.handlers:
#     handler.setLevel(logging.INFO)
#     handler.setFormatter(logging.Formatter('%(message)s', None, 'utf-8'))

model = YOLO("yolov8n.pt")
# Use the model # datasets/dataset/images/
if __name__ == '__main__':
    # Use the model
    results = model.train(data='/Users/hong/Desktop/Project/psycho/su_detect/class_detect2/datasets/dataset/data.yaml', epochs=50, batch=4, val=True)

    # success = model.export(format='onnx')



