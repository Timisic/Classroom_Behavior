# coding:utf-8
from ultralytics import YOLO
import cv2

# 所需加载的模型目录
# path = 'models/best.pt'
path = 'runs/detect/train9/weights/best.pt'
# 需要检测的图片地址
img_path = "/Users/hong/Desktop/Project/psycho/su_detect/img/WechatIMG126.jpg"

# 加载预训练模型
# conf	0.25	object confidence threshold for detection
# iou	0.7	intersection over union (IoU) threshold for NMS
model = YOLO(path, task='detect')
# model = YOLO(path, task='detect',conf=0.5)

# 检测图片
results = model(img_path)
detections = results[0]  # 获取检测结果

# 类别标签
class_names = ['down', 'lookaround', 'phone', 'up']
class_counts = {class_name: 0 for class_name in class_names}
total_count = 0

# 解析检测结果
for det in detections.boxes:
    class_id = int(det.cls)
    class_name = class_names[class_id]
    class_counts[class_name] += 1
    total_count += 1

# 打印结果
print("总人数:", total_count)
for class_name, count in class_counts.items():
    print(f"{class_name} 人数: {count}")


# 绘制检测结果不显示标签
def plot_boxes_no_labels(results, img):
    colors = {
        'down': (0, 0, 255),  # 红色
        'lookaround': (255, 255, 0),  # 黄色
        'phone': (255, 0, 0),       # 蓝色
        'up': (0, 255, 0)  # 绿色
    }

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 获取边界框坐标
        class_id = int(box.cls)
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

# # 检测图片，带标签
# results = model(img_path)
# print(results)
# res = results[0].plot()
# res = cv2.resize(res, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
# cv2.imshow("YOLOv8 Detection", res)
# cv2.waitKey(0)
