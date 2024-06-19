# -*- coding: utf-8 -*-
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QWidget, QHeaderView, QTableWidgetItem, \
    QAbstractItemView, QVBoxLayout, QLabel
import sys
import os
from PIL import ImageFont
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

sys.path.append('UIProgram')
from UIProgram.UiMain_t3 import Ui_MainWindow
import sys
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QCoreApplication
import detect_tools as tools
import cv2
import Config
import img_heatmap
from UIProgram.QssLoader import QSSLoader
from UIProgram.precess_bar import ProgressBar
import numpy as np


# import torch


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.initMain()
        self.signalconnect()
        self.model_path = 'runs/detect/train9/weights/best.pt'
        self.label_show_4 = self.findChild(QLabel, 'label_show_4')
        self.label_show_5 = self.findChild(QLabel, 'label_show_5')
        self.label_show_6 = self.findChild(QLabel, 'label_show_6')
        self.label_show_7 = self.findChild(QLabel, 'label_show_7')
        self.label_show_8 = self.findChild(QLabel, 'label_show_8')
        self.label_show_9 = self.findChild(QLabel, 'label_show_9')
        self.label_show_10 = self.findChild(QLabel, 'label_show_10')
        self.label_show_11 = self.findChild(QLabel, 'label_show_11')

        self.labels = [
            self.label_show_4,
            self.label_show_5,
            self.label_show_6,
            self.label_show_7,
            self.label_show_8,
            self.label_show_9,
            self.label_show_10,
            self.label_show_11,
        ]

        self.probs_over_time = []  # 用于存储每隔 x 秒的 probs 值
        self.time_intervals = []  # 用于存储时间间隔
        self.frame_interval = 30  # 自定义跳帧数，例如每隔 30 帧记录一次 prob
        self.current_time = 0  # 用于记录当前时间.

        # 初始化 Matplotlib 图像
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ui.plot_layout = QVBoxLayout(self.ui.label_show_2)
        self.ui.plot_layout.addWidget(self.canvas)
        self.ui.label_show_2.setLayout(self.ui.plot_layout)
        self.ui.label_show_2.setScaledContents(True)

        # 闪烁功能
        # 闪烁定时器
        self.active_labels = None
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self.blink)
        self.blink_state = False
        self.blink_interval = 60  # 闪烁间隔
        # 状态检测定时器
        self.state_check_timer = QTimer()
        self.state_check_timer.timeout.connect(self.blinkstate)
        self.state_check_timer.start(40)

        # 加载css渲染效果
        style_file = 'UIProgram/style.css'
        qssStyleSheet = QSSLoader.read_qss_file(style_file)
        self.setStyleSheet(qssStyleSheet)

    def signalconnect(self):
        self.ui.PicBtn.clicked.connect(self.open_img)
        self.ui.comboBox.activated.connect(self.combox_change)
        self.ui.VideoBtn.clicked.connect(self.vedio_show)
        self.ui.CapBtn.clicked.connect(self.camera_show)
        self.ui.SaveBtn.clicked.connect(self.save_detect_video)
        self.ui.ExitBtn.clicked.connect(QCoreApplication.quit)
        self.ui.FilesBtn.clicked.connect(self.detact_batch_imgs)

    def initMain(self):
        self.show_width = 770
        self.show_height = 480

        self.org_path = None

        self.is_camera_open = False
        self.cap = None

        # self.device = 0 if torch.cuda.is_available() else 'cpu'

        # 加载检测模型
        self.model = YOLO(Config.model_path, task='detect')
        self.model(np.zeros((48, 48, 3)))  # 预先加载推理模型
        self.fontC = ImageFont.truetype("Font/platech.ttf", 25, 0)

        # 用于绘制不同颜色矩形框
        self.colors = tools.Colors()

        # 更新视频图像
        self.timer_camera = QTimer()

        # 更新检测信息表格
        # self.timer_info = QTimer()
        # 保存视频
        self.timer_save_video = QTimer()

        # 表格
        self.ui.tableWidget.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.ui.tableWidget.verticalHeader().setDefaultSectionSize(40)
        self.ui.tableWidget.setColumnWidth(0, 80)  # 设置列宽
        self.ui.tableWidget.setColumnWidth(1, 200)
        self.ui.tableWidget.setColumnWidth(2, 150)
        self.ui.tableWidget.setColumnWidth(3, 90)
        self.ui.tableWidget.setColumnWidth(4, 230)
        # self.ui.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 表格铺满
        # self.ui.tableWidget.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        # self.ui.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)  # 设置表格不可编辑
        self.ui.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)  # 设置表格整行选中
        self.ui.tableWidget.verticalHeader().setVisible(False)  # 隐藏列标题
        self.ui.tableWidget.setAlternatingRowColors(True)  # 表格背景交替

    def open_img(self):
        """
        img目标检测
        """
        if self.cap:
            # 打开图片前关闭摄像头
            self.video_stop()
            self.is_camera_open = False
            self.ui.CaplineEdit.setText('摄像头未开启')
            self.cap = None

        # 弹出的窗口名称：'打开图片'
        # 默认打开的目录：'./'
        # 只能打开.jpg与.gif结尾的图片文件
        # file_path, _ = QFileDialog.getOpenFileName(self.ui.centralwidget, '打开图片', './', "Image files (*.jpg *.gif)")
        file_path, _ = QFileDialog.getOpenFileName(None, '打开图片', './', "Image files (*.jpg *.jpeg *.png)")
        if not file_path:
            return

        self.ui.comboBox.setDisabled(False)
        self.org_path = file_path
        self.org_img = tools.img_cvread(self.org_path)

        # 调用 detect_and_create_heatmap 函数
        overlay, total_count, class_counts = img_heatmap.detect_and_create_heatmap(self.org_path)

        # 打印结果
        # print("总人数:", total_count)
        # for class_name, count in class_counts.items():
        #     print(f"{class_name} 人数: {count}")

        # 目标检测
        t1 = time.time()
        self.results = self.model(self.org_path)[0]
        t2 = time.time()
        take_time_str = '{:.3f} s'.format(t2 - t1)
        self.ui.time_lb.setText(take_time_str)

        location_list = self.results.boxes.xyxy.tolist()
        self.location_list = [list(map(int, e)) for e in location_list]
        cls_list = self.results.boxes.cls.tolist()
        self.cls_list = [int(i) for i in cls_list]
        self.conf_list = self.results.boxes.conf.tolist()
        self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]

        total_nums = len(location_list)
        cls_percents = []
        for i in range(6):
            res = self.cls_list.count(i) / total_nums
            cls_percents.append(res)
        self.set_percent(cls_percents)

        now_img = self.results.plot()

        self.draw_img = now_img
        # 获取缩放后的图片尺寸
        self.img_width, self.img_height = self.get_resize_size(now_img)
        resize_cvimg = cv2.resize(overlay, (self.img_width, self.img_height))  # 使用 overlay 代替 now_img
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show.setPixmap(pix_img)
        self.ui.label_show.setAlignment(Qt.AlignCenter)

        # 设置路径显示
        self.ui.PiclineEdit.setText(self.org_path)

        # 目标数目
        target_nums = len(self.cls_list)
        self.ui.label_nums.setText(str(target_nums))

        # 设置目标选择下拉框
        choose_list = ['全部']
        target_names = [Config.names[id] + '_' + str(index) for index, id in enumerate(self.cls_list)]
        choose_list = choose_list + target_names

        self.ui.comboBox.clear()
        self.ui.comboBox.addItems(choose_list)

        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()
        self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=self.org_path)

    def detact_batch_imgs(self):
        if self.cap:
            # 打开图片前关闭摄像头
            self.video_stop()
            self.is_camera_open = False
            self.ui.CaplineEdit.setText('摄像头未开启')
            self.cap = None
        directory = QFileDialog.getExistingDirectory(self,
                                                     "选取文件夹",
                                                     "./")  # 起始路径
        if not directory:
            return
        self.org_path = directory
        img_suffix = ['jpg', 'png', 'jpeg', 'bmp']
        for file_name in os.listdir(directory):
            full_path = os.path.join(directory, file_name)
            if os.path.isfile(full_path) and file_name.split('.')[-1].lower() in img_suffix:
                # self.ui.comboBox.setDisabled(False)
                img_path = full_path
                self.org_img = tools.img_cvread(img_path)
                # 目标检测
                t1 = time.time()
                self.results = self.model(img_path)[0]
                t2 = time.time()
                take_time_str = '{:.3f} s'.format(t2 - t1)
                self.ui.time_lb.setText(take_time_str)

                location_list = self.results.boxes.xyxy.tolist()
                self.location_list = [list(map(int, e)) for e in location_list]
                cls_list = self.results.boxes.cls.tolist()
                self.cls_list = [int(i) for i in cls_list]
                self.conf_list = self.results.boxes.conf.tolist()
                self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]

                total_nums = len(location_list)
                cls_percents = []
                for i in range(6):
                    res = self.cls_list.count(i) / total_nums
                    cls_percents.append(res)
                self.set_percent(cls_percents)

                now_img = self.results.plot()
                # 调用 detect_and_create_heatmap 函数
                overlay, total_count, class_counts = img_heatmap.detect_and_create_heatmap(self.org_path)
                self.draw_img = now_img
                # 获取缩放后的图片尺寸
                self.img_width, self.img_height = self.get_resize_size(now_img)
                # resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
                resize_cvimg = cv2.resize(overlay, (self.img_width, self.img_height))  # 使用 overlay 代替 now_img
                pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
                self.ui.label_show.setPixmap(pix_img)
                self.ui.label_show.setAlignment(Qt.AlignCenter)

                # 设置路径显示
                self.ui.PiclineEdit.setText(img_path)

                # 目标数目
                target_nums = len(self.cls_list)
                self.ui.label_nums.setText(str(target_nums))

                # 设置目标选择下拉框
                choose_list = ['全部']
                target_names = [Config.names[id] + '_' + str(index) for index, id in enumerate(self.cls_list)]
                choose_list = choose_list + target_names

                self.ui.comboBox.clear()
                self.ui.comboBox.addItems(choose_list)
                self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=img_path)
                self.ui.tableWidget.scrollToBottom()
                QApplication.processEvents()  # 刷新页面

    # def draw_rect_and_tabel(self, results, img):
    #     now_img = img.copy()
    #     location_list = results.boxes.xyxy.tolist()
    #     self.location_list = [list(map(int, e)) for e in location_list]
    #     cls_list = results.boxes.cls.tolist()
    #     self.cls_list = [int(i) for i in cls_list]
    #     self.conf_list = results.boxes.conf.tolist()
    #     self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]
    #
    #     for loacation, type_id, conf in zip(self.location_list, self.cls_list, self.conf_list):
    #         type_id = int(type_id)
    #         color = self.colors(int(type_id), True)
    #         # cv2.rectangle(now_img, (int(x1), int(y1)), (int(x2), int(y2)), colors(int(type_id), True), 3)
    #         now_img = tools.drawRectBox(now_img, loacation, Config.CH_names[type_id], self.fontC, color)
    #
    #     # 获取缩放后的图片尺寸
    #     # 调用 detect_and_create_heatmap 函数
    #     overlay, total_count, class_counts = img_heatmap.detect_and_create_heatmap(self.org_path,
    #                                                                                self.model_path)
    #     self.img_width, self.img_height = self.get_resize_size(now_img)
    #     # resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
    #     resize_cvimg = cv2.resize(overlay, (self.img_width, self.img_height))
    #     pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
    #     self.ui.label_show.setPixmap(pix_img)
    #     self.ui.label_show.setAlignment(Qt.AlignCenter)
    #
    #     # 设置路径显示
    #     self.ui.PiclineEdit.setText(self.org_path)
    #     # 目标数目
    #     target_nums = len(self.cls_list)
    #     self.ui.label_nums.setText(str(target_nums))
    #     # 删除表格所有行
    #     self.ui.tableWidget.setRowCount(0)
    #     self.ui.tableWidget.clearContents()
    #     self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=self.org_path)
    #     return now_img

    def combox_change(self):
        com_text = self.ui.comboBox.currentText()
        if com_text == '全部':
            cur_box = self.location_list
            cur_img = self.results.plot()
            self.ui.type_lb.setText(Config.CH_names[self.cls_list[0]])
            self.ui.label_conf.setText(str(self.conf_list[0]))
        else:
            index = int(com_text.split('_')[-1])
            cur_box = [self.location_list[index]]
            cur_img = self.results[index].plot()
            self.ui.type_lb.setText(Config.CH_names[self.cls_list[index]])
            self.ui.label_conf.setText(str(self.conf_list[index]))

        # 调用 detect_and_create_heatmap 函数
        overlay, total_count, class_counts = img_heatmap.detect_and_create_heatmap(self.org_path)
        # resize_cvimg = cv2.resize(cur_img, (self.img_width, self.img_height))
        resize_cvimg = cv2.resize(overlay, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show.clear()
        self.ui.label_show.setPixmap(pix_img)
        self.ui.label_show.setAlignment(Qt.AlignCenter)

    def get_video_path(self):
        file_path, _ = QFileDialog.getOpenFileName(None, '打开视频', './', "Image files (*.avi *.mp4 *.jepg *.png)")
        if not file_path:
            return None
        self.org_path = file_path
        self.ui.VideolineEdit.setText(file_path)
        return file_path

    def video_start(self):
        # 删除表格所有行
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()
        # 清空下拉框
        self.ui.comboBox.clear()
        # 定时器开启，每隔一段时间，读取一帧
        self.timer_camera.start(1)
        self.timer_camera.timeout.connect(self.open_frame)

    def tabel_info_show(self, locations, clses, confs, path=None):
        path = path
        for location, cls, conf in zip(locations, clses, confs):
            row_count = self.ui.tableWidget.rowCount()  # 返回当前行数(尾部)
            self.ui.tableWidget.insertRow(row_count)  # 尾部插入一行
            item_id = QTableWidgetItem(str(row_count + 1))  # 序号
            item_id.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中
            item_path = QTableWidgetItem(str(path))  # 路径
            # item_path.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

            item_cls = QTableWidgetItem(str(Config.CH_names[cls]))
            item_cls.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中

            item_conf = QTableWidgetItem(str(conf))
            item_conf.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中

            item_location = QTableWidgetItem(str(location))  # 目标框位置
            # item_location.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中

            self.ui.tableWidget.setItem(row_count, 0, item_id)
            self.ui.tableWidget.setItem(row_count, 1, item_path)
            self.ui.tableWidget.setItem(row_count, 2, item_cls)
            self.ui.tableWidget.setItem(row_count, 3, item_conf)
            self.ui.tableWidget.setItem(row_count, 4, item_location)
        self.ui.tableWidget.scrollToBottom()

    def video_stop(self):
        self.cap.release()
        self.timer_camera.stop()
        # self.timer_info.stop()

    def open_frame(self):
        ret, now_img = self.cap.read()
        if ret:
            # 设置跳帧
            self.current_time += 1
            if self.current_time % self.frame_interval != 0:
                return

            # 目标检测
            t1 = time.time()
            results = self.model(now_img)[0]
            t2 = time.time()
            take_time_str = '{:.3f} s'.format(t2 - t1)
            self.ui.time_lb.setText(take_time_str)

            location_list = results.boxes.xyxy.tolist()
            self.location_list = [list(map(int, e)) for e in location_list]
            cls_list = results.boxes.cls.tolist()
            self.cls_list = [int(i) for i in cls_list]
            self.conf_list = results.boxes.conf.tolist()
            self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]

            total_nums = len(location_list)
            cls_percents = []
            for i in range(6):
                if total_nums != 0:  # 识别到的目标数量
                    res = self.cls_list.count(i) / total_nums
                else:
                    res = 0
                cls_percents.append(res)
            self.set_percent(cls_percents)

            if self.current_time % self.frame_interval == 0:
                self.probs_over_time.append(cls_percents[:4])  # 假设只记录前4个class的概率
                self.time_intervals.append(self.current_time / 30)  # 假设视频帧率为 30 fps
                self.plot_probs_over_time()

            # now_img = results.plot()

            # 获取缩放后的图片尺寸
            self.img_width, self.img_height = self.get_resize_size(now_img)

            # 调用 heatmap 函数
            overlay, total_count, class_counts = img_heatmap.detect_and_create_heatmap(now_img, is_path=False)
            # resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
            resize_cvimg = cv2.resize(overlay, (self.img_width, self.img_height))
            pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
            self.ui.label_show.setPixmap(pix_img)
            self.ui.label_show.setAlignment(Qt.AlignCenter)

            # 目标数目
            target_nums = len(self.cls_list)
            self.ui.label_nums.setText(str(target_nums))

            # 设置目标选择下拉框
            choose_list = ['全部']
            target_names = [Config.names[id] + '_' + str(index) for index, id in enumerate(self.cls_list)]
            choose_list = choose_list + target_names

            self.ui.comboBox.clear()
            self.ui.comboBox.addItems(choose_list)
            self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=self.org_path)
        else:
            self.cap.release()
            self.timer_camera.stop()

    def vedio_show(self):
        if self.is_camera_open:
            self.is_camera_open = False
            self.ui.CaplineEdit.setText('摄像头未开启')

        video_path = self.get_video_path()
        if not video_path:
            return None
        self.cap = cv2.VideoCapture(video_path)
        self.video_start()
        self.ui.comboBox.setDisabled(True)

    def camera_show(self):
        self.is_camera_open = not self.is_camera_open
        if self.is_camera_open:
            self.ui.CaplineEdit.setText('摄像头开启')
            self.cap = cv2.VideoCapture(0)
            self.video_start()
            self.ui.comboBox.setDisabled(True)
        else:
            self.ui.CaplineEdit.setText('摄像头未开启')
            self.ui.label_show.setText('')
            if self.cap:
                self.cap.release()
                cv2.destroyAllWindows()
            self.ui.label_show.clear()

    def set_percent(self, probs):

        # 显示各行为概率值
        items = [self.ui.progressBar, self.ui.progressBar_2, self.ui.progressBar_3, self.ui.progressBar_4, ]
        labels = [self.ui.label_20, self.ui.label_21, self.ui.label_22, self.ui.label_23]

        if len(probs) > len(labels):
            probs = probs[:len(labels)]
        else:
            labels = labels[:len(probs)]

        prob_values = [round(each * 100) for each in probs]
        label_values = ['{:.1f}%'.format(each * 100) for each in probs]

        for i in range(len(probs)):
            items[i].setValue(prob_values[i])
            labels[i].setText(label_values[i])

        if probs:
            self.update_light(probs[0])

    def get_resize_size(self, img):
        _img = img.copy()
        img_height, img_width, depth = _img.shape
        ratio = img_width / img_height
        if ratio >= self.show_width / self.show_height:
            self.img_width = self.show_width
            self.img_height = int(self.img_width / ratio)
        else:
            self.img_height = self.show_height
            self.img_width = int(self.img_height * ratio)
        return self.img_width, self.img_height

    def save_detect_video(self):
        if self.cap is None and not self.org_path:
            QMessageBox.about(self, '提示', '当前没有可保存信息，请先打开图片或视频！')
            return

        if self.is_camera_open:
            QMessageBox.about(self, '提示', '摄像头视频无法保存!')
            return

        if self.cap:
            res = QMessageBox.information(self, '提示', '保存视频检测结果可能需要较长时间，请确认是否继续保存？',
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if res == QMessageBox.Yes:
                self.video_stop()
                com_text = self.ui.comboBox.currentText()
                self.btn2Thread_object = btn2Thread(self.org_path, self.model, com_text)
                self.btn2Thread_object.start()
                self.btn2Thread_object.update_ui_signal.connect(self.update_process_bar)
            else:
                return
        else:
            if os.path.isfile(self.org_path):
                fileName = os.path.basename(self.org_path)
                name, end_name = fileName.rsplit(".", 1)
                save_name = name + '_detect_result.' + end_name
                save_img_path = os.path.join(Config.save_path, save_name)
                # 保存图片
                cv2.imwrite(save_img_path, self.draw_img)
                QMessageBox.about(self, '提示', '图片保存成功!\n文件路径:{}'.format(save_img_path))
            else:
                img_suffix = ['jpg', 'png', 'jpeg', 'bmp']
                for file_name in os.listdir(self.org_path):
                    full_path = os.path.join(self.org_path, file_name)
                    if os.path.isfile(full_path) and file_name.split('.')[-1].lower() in img_suffix:
                        name, end_name = file_name.rsplit(".", 1)
                        save_name = name + '_detect_result.' + end_name
                        save_img_path = os.path.join(Config.save_path, save_name)
                        results = self.model(full_path)[0]
                        now_img = results.plot()
                        # 保存图片
                        cv2.imwrite(save_img_path, now_img)

                QMessageBox.about(self, '提示', '图片保存成功!\n文件路径:{}'.format(Config.save_path))

    def update_process_bar(self, cur_num, total):
        if cur_num == 1:
            self.progress_bar = ProgressBar(self)
            self.progress_bar.show()
        if cur_num >= total:
            self.progress_bar.close()
            QMessageBox.about(self, '提示', '视频保存成功!\n文件在{}目录下'.format(Config.save_path))
            return
        if self.progress_bar.isVisible() is False:
            # 点击取消保存时，终止进程
            self.btn2Thread_object.stop()
            return
        value = int(cur_num / total * 100)
        self.progress_bar.setValue(cur_num, total, value)
        QApplication.processEvents()

    def plot_probs_over_time(self):
        fixed_width = 7.7  # 英寸
        fixed_height = 2  # 英寸

        # 调整 FigureCanvas 的大小以适应 QLabel
        self.figure.set_size_inches(fixed_width, fixed_height)
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # 设置固定的坐标范围，根据视频长短指定
        # 此处设定分钟数
        min = 3
        min2s = min * 60
        # 单位s
        ax.set_xlim(0, min2s)
        ax.set_ylim(0, 1)

        probs_array = np.array(self.probs_over_time)
        for i in range(4):
            ax.plot(self.time_intervals, probs_array[:, i], label=f'Class {i}')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Probability')
        ax.legend(loc='upper right')
        ax.set_title('Class Probabilities Over Time')

        # 使用紧凑布局确保所有内容都能显示
        self.figure.tight_layout()
        self.canvas.draw()

    def update_light(self, rate):
        # 阈值设置
        if rate < 0.1:
            self.active_labels = 2
        elif rate < 0.2:
            self.active_labels = 4
        elif rate < 0.4:
            self.active_labels = 6
        else:
            self.active_labels = 8

        """
        这边会重复进行计算，
        耗时且大概闪烁问题出现在此
        """
        for i in range(8):
            if i < self.active_labels:
                # 计算颜色的渐变，绿色(0, 255, 0)到红色(255, 0, 0)
                green_value = int(255 * (1 - (i / 7)))
                red_value = int(255 * (i / 7))
                color = f"rgb({red_value}, {green_value}, 0)"
                self.labels[i].setStyleSheet(f"background-color: {color}; border: 1px solid black;")
            else:
                self.labels[i].setStyleSheet("background-color: transparent; border: 1px solid transparent;")

    """
    目前还做不到闪烁独立于检测why？
    但视频运行停止后，闪烁会一直持续（设定但间隔正常）
    """
    def blinkstate(self):
        # 达到最高阈值，启动或保持闪烁定时器
        if self.active_labels == 8:
            if not self.blink_timer.isActive():
                self.blink_timer.start(self.blink_interval)
        else:
            self.blink_timer.stop()
            self.blink_state = False
            self.labels[-1].setStyleSheet("background-color: transparent; border: 1px solid transparent;")

    def blink(self):
        # 闪烁
        self.blink_state = not self.blink_state
        color = "rgb(255, 0, 0)" if self.blink_state else "transparent"
        self.labels[-1].setStyleSheet(
            f"background-color: {color}; border: 1px solid transparent;" if color == "transparent" else f"background-color: {color}; border: 1px solid black;"
        )


class btn2Thread(QThread):
    """
    进行检测后的视频保存
    """
    # 声明一个信号
    update_ui_signal = pyqtSignal(int, int)

    def __init__(self, path, model, com_text):
        super(btn2Thread, self).__init__()
        self.org_path = path
        self.model = model
        self.com_text = com_text
        # 用于绘制不同颜色矩形框
        self.colors = tools.Colors()
        self.is_running = True  # 标志位，表示线程是否正在运行

    def run(self):
        # VideoCapture方法是cv2库提供的读取视频方法
        cap = cv2.VideoCapture(self.org_path)
        # 设置需要保存视频的格式“xvid”
        # 该参数是MPEG-4编码类型，文件名后缀为.avi
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # 设置视频帧频
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 设置视频大小
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # VideoWriter方法是cv2库提供的保存视频方法
        # 按照设置的格式来out输出
        fileName = os.path.basename(self.org_path)
        name, end_name = fileName.split('.')
        save_name = name + '_detect_result.avi'
        save_video_path = os.path.join(Config.save_path, save_name)
        out = cv2.VideoWriter(save_video_path, fourcc, fps, size)

        prop = cv2.CAP_PROP_FRAME_COUNT
        total = int(cap.get(prop))
        print("[INFO] 视频总帧数：{}".format(total))
        cur_num = 0

        # 确定视频打开并循环读取
        while (cap.isOpened() and self.is_running):
            cur_num += 1
            print('当前第{}帧，总帧数{}'.format(cur_num, total))
            # 逐帧读取，ret返回布尔值
            # 参数ret为True 或者False,代表有没有读取到图片
            # frame表示截取到一帧的图片
            ret, frame = cap.read()
            if ret == True:
                # 检测
                results = self.model(frame)[0]
                frame = results.plot()
                out.write(frame)
                self.update_ui_signal.emit(cur_num, total)
            else:
                break
        # 释放资源
        cap.release()
        out.release()

    def stop(self):
        self.is_running = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
