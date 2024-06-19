# coding:utf-8

# 图片及视频检测结果保存路径
save_path = 'save_data'

# 使用的模型路径
model_path = 'runs/detect/train9/weights/best.pt'
names = {0: 'down', 1: 'lookaround', 2: 'phone', 3: 'up'}
# # class names
# names:  [ 'hand-raising', 'reading', 'writing','using phone', 'bowing the head', 'leaning over the table']
CH_names = ['低头', '环顾', '手机', '抬头']
