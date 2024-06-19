# coding:utf-8
import cv2
from ultralytics import YOLO
from sahi.predict import get_sliced_prediction
from sahi.utils.yolov8 import download_yolov8s_model
from sahi import AutoDetectionModel
import torch

# 所需加载的模型目录
path = 'runs/detect/train9/weights/best.pt'
# 需要检测的图片地址
img_path = "/Users/hong/Desktop/Project/psycho/su_detect/img/img3.jpg"

# 加载预训练YOLOv8模型
model = YOLO(path, task='detect')

# 使用SAHI包装YOLOv8模型
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=path,
    confidence_threshold=0.15,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)

# 检测图片，使用SAHI进行切片预测
result = get_sliced_prediction(
    img_path,
    detection_model,
    slice_height=350,
    slice_width=350,
    overlap_height_ratio=0.15,
    overlap_width_ratio=0.15,
    # perform_standard_pred=True,  # 启用多尺度推理
)

# 获取检测结果
detections = result.object_prediction_list

# 类别标签
class_names = ['down', 'lookaround', 'phone', 'up']
class_counts = {class_name: 0 for class_name in class_names}
total_count = 0

# 解析检测结果
for det in detections:
    class_id = int(det.category.id)
    class_name = class_names[class_id]
    class_counts[class_name] += 1
    total_count += 1

# 打印结果
print("总人数:", total_count)
for class_name, count in class_counts.items():
    print(f"{class_name} 人数: {count}")


# 绘制检测结果不显示标签
def plot_boxes_no_labels(detections, img):
    colors = {
        'down': (0, 0, 255),  # 红色
        'lookaround': (255, 255, 0),  # 黄色
        'phone': (255, 0, 0),  # 蓝色
        'up': (0, 255, 0)  # 绿色
    }

    for det in detections:
        x1, y1, x2, y2 = map(int, [det.bbox.minx, det.bbox.miny, det.bbox.maxx, det.bbox.maxy])  # 获取边界框坐标
        class_id = int(det.category.id)
        class_name = class_names[class_id]
        color = colors[class_name]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


# 获取检测结果
img = cv2.imread(img_path)
plot_boxes_no_labels(detections, img)

# 调整图像大小并显示
res = cv2.resize(img, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
cv2.imshow("YOLOv8 Detection", res)
cv2.waitKey(0)
cv2.destroyAllWindows()
