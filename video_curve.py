# coding:utf-8
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import time

# 所需加载的模型目录
path = 'runs/detect/train9/weights/best.pt'
# 需要检测的视频地址
video_path = "/Users/hong/Desktop/Project/psycho/su_detect/img/课堂.mov"
# 设置保存输出视频和图片的路径
output_video_path = "/Users/hong/Desktop/Project/psycho/su_detect/output/annotated_video.mp4"
output_image_path = "/Users/hong/Desktop/Project/psycho/su_detect/output/down_count_plot.png"

# 加载YOLOv8模型
model = YOLO(path)
cap = cv2.VideoCapture(video_path)

# 获取视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    raise ValueError("无法获取视频的帧率")

# 初始化视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 初始化计数变量和时间变量
frame_count = 0
down_counts = []
timestamps = []

# 设置跳帧数和时间窗口
frame_skip = 12  # 尽量处理每一帧，以便与现实时间同步
time_window = 4  # 每5秒记录一次

# 定义颜色
class_names = ['down', 'lookaround', 'phone', 'up']
colors = {
    'down': (0, 0, 255),  # 红色
    'lookaround': (255, 255, 0),  # 黄色
    'phone': (255, 0, 0),  # 蓝色
    'up': (0, 255, 0)  # 绿色
}

# Loop through the video frames
start_time = time.time()
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if not success:
        # Break the loop if the end of the video is reached
        break

    # 跳过指定帧数
    if frame_count % frame_skip == 0:
        # Resize the frame to speed up processing
        frame_resized = cv2.resize(frame, (640, 360))

        # Run YOLOv8 inference on the frame
        start_inference = time.time()
        results = model(frame_resized)
        inference_time = time.time() - start_inference

        # Count 'down' class instances
        down_count = 0
        for det in results[0].boxes:
            class_id = int(det.cls)
            if class_id == 0:  # Assuming 'down' is the first class in class_names
                down_count += 1

            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            class_name = 'down' if class_id == 0 else 'up' if class_id == 2 else class_names[class_id]
            color = colors[class_name]
            # 只绘制颜色框，不显示标签
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)

        # Append counts and timestamps
        down_counts.append(down_count)
        timestamps.append(frame_count / fps)

        # Resize back to original size for saving
        frame_annotated = cv2.resize(frame_resized, (width, height))
        out.write(frame_annotated)  # Save the frame to the output video

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", frame_resized)

        # 控制帧率，使推理时间与视频帧率相匹配
        elapsed_time = time.time() - start_time
        expected_time = frame_count / fps
        if elapsed_time < expected_time:
            time.sleep(expected_time - elapsed_time)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_count += 1

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()

# 将down_counts和timestamps转换为numpy数组以便于计算
down_counts = np.array(down_counts)
timestamps = np.array(timestamps)

# 计算步长，确保步长不为零
step = max(1, int(time_window * fps / frame_skip))

# 平滑曲线 - 每time_window秒取一次平均值
smoothed_down_counts = []
smoothed_timestamps = []

for i in range(0, len(timestamps), step):
    window_counts = down_counts[i:i + step]
    if len(window_counts) > 0:
        smoothed_down_counts.append(np.mean(window_counts))
        smoothed_timestamps.append(timestamps[i])

# 绘制平滑后的'down'人数随时间变化的曲线
plt.plot(smoothed_timestamps, smoothed_down_counts, label='Down Count')
plt.xlabel('Time (s)')
plt.ylabel('Count')
plt.title('Down Count Over Time')
plt.legend()
# 保存图像
plt.savefig(output_image_path)
plt.show()
