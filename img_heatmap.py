import cv2
import numpy as np
from ultralytics import YOLO
from sahi.predict import get_sliced_prediction
from sahi.utils.yolov8 import download_yolov8s_model
from sahi import AutoDetectionModel
import torch
from PIL import Image

class_names = ['down', 'lookaround', 'phone', 'up']

def detect_and_create_heatmap(img_input, model_path='runs/detect/train9/weights/best.pt', is_path=True):
    # 加载预训练YOLOv8模型
    model = YOLO(model_path, task='detect')

    # 使用SAHI包装YOLOv8模型
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=0.15,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    # 如果传入的是路径，使用SAHI进行切片预测
    if is_path:
        result = get_sliced_prediction(
            img_input,
            detection_model,
            # slice_height=350,
            # slice_width=350,
            # overlap_height_ratio=0.15,
            # overlap_width_ratio=0.15,
            slice_height=500,
            slice_width=500,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.1,
        )
        img = cv2.imread(img_input)
    else:
        # 将图像数据转换为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))
        result = get_sliced_prediction(
            img_pil,
            detection_model,
            slice_height=500,
            slice_width=500,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.1,
        )
        img = img_input

    # 获取检测结果
    detections = result.object_prediction_list

    # 类别标签
    class_counts = {class_name: 0 for class_name in class_names}
    total_count = 0

    # 创建热图
    heatmap_down = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    heatmap_up = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

    # 解析检测结果
    for det in detections:
        x1, y1, x2, y2 = map(int, [det.bbox.minx, det.bbox.miny, det.bbox.maxx, det.bbox.maxy])
        class_id = int(det.category.id)
        class_name = class_names[class_id]
        class_counts[class_name] += 1
        total_count += 1

        # 更新热图（低头使用暖色调，抬头使用冷色调）
        if class_name == 'down':
            heatmap_down[y1:y2, x1:x2] += 1
        elif class_name == 'up':
            heatmap_up[y1:y2, x1:x2] += 1

    # 应用高斯模糊来扩散热图
    heatmap_down = cv2.GaussianBlur(heatmap_down, (0, 0), sigmaX=15, sigmaY=15)
    heatmap_up = cv2.GaussianBlur(heatmap_up, (0, 0), sigmaX=5, sigmaY=5)

    # 归一化热图
    heatmap_down = cv2.normalize(heatmap_down, heatmap_down, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    heatmap_down = np.uint8(heatmap_down)
    heatmap_up = cv2.normalize(heatmap_up, heatmap_up, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    heatmap_up = np.uint8(heatmap_up)

    # 创建掩膜
    mask_down = heatmap_down > 0
    mask_up = heatmap_up > 0

    # 应用颜色映射（暖色调和冷色调）
    colored_heatmap_down = cv2.applyColorMap(heatmap_down, cv2.COLORMAP_JET)  # 暖色调
    colored_heatmap_up = cv2.applyColorMap(heatmap_up, cv2.COLORMAP_OCEAN)  # 冷色调

    # 遮盖无数据部分
    colored_heatmap_down[~mask_down] = 0
    colored_heatmap_up[~mask_up] = 0

    # 叠加热图到原始图像
    overlay = cv2.addWeighted(img, 1.0, colored_heatmap_down, 0.6, 0)
    overlay = cv2.addWeighted(overlay, 1.0, colored_heatmap_up, 0.6, 0)

    return overlay, total_count, class_counts

if __name__ == "__main__":
    img_path = "/Users/hong/Desktop/Project/psycho/su_detect/img/img3.jpg"
    model_path = 'runs/detect/train9/weights/best.pt'

    overlay, total_count, class_counts = detect_and_create_heatmap(img_path, model_path)

    # 打印结果
    print("总人数:", total_count)
    for class_name, count in class_counts.items():
        print(f"{class_name} 人数: {count}")

    # 显示结果
    res = cv2.resize(overlay, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("YOLOv8 Detection with Heatmap", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
