#!/usr/bin/env python3
"""
YOLOv8 + 头部深度相机测距（无图形界面版）
- 使用头部彩色相机（kHeadColor）进行目标检测
- 使用头部深度相机（kHeadDepth）获取距离
- 结果保存为图像到 output_frames 目录，同时打印距离信息到终端
"""

import time
import cv2
import numpy as np
from ultralytics import YOLO
import agibot_gdk
import os

class YOLOWithDistance:
    def __init__(self):
        # 初始化YOLO模型（使用nano版本，可根据需要更换）
        self.model = YOLO("yolov8n.pt")
        
        # 初始化相机
        print("正在初始化相机...")
        self.camera = agibot_gdk.Camera()
        time.sleep(3)  # 等待相机稳定
        
        # 定义使用的相机类型
        self.color_camera_type = agibot_gdk.CameraType.kHeadColor
        self.depth_camera_type = agibot_gdk.CameraType.kHeadDepth
        
        # 验证相机是否可用
        self._check_camera()
        
        # 创建输出目录
        os.makedirs("output_frames", exist_ok=True)
        print(f"初始化完成！结果将保存到 {os.path.abspath('output_frames')} 目录")
    
    def _check_camera(self):
        """快速验证相机是否能获取图像"""
        color_img = self.camera.get_latest_image(self.color_camera_type, 1000.0)
        depth_img = self.camera.get_latest_image(self.depth_camera_type, 1000.0)
        if color_img is None or depth_img is None:
            raise RuntimeError("无法获取头部彩色或深度图像，请检查相机连接")
        print(f"彩色图尺寸: {color_img.width}x{color_img.height}, 深度图尺寸: {depth_img.width}x{depth_img.height}")
    
    def decode_image(self, image):
        """
        将SDK图像对象解码为OpenCV格式的numpy数组
        支持JPEG压缩和多种未压缩格式
        """
        if image.encoding == agibot_gdk.Encoding.JPEG:
            nparr = np.frombuffer(image.data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        elif image.encoding == agibot_gdk.Encoding.UNCOMPRESSED:
            # 根据颜色格式转换
            if image.color_format == agibot_gdk.ColorFormat.RGB:
                img = np.frombuffer(image.data, dtype=np.uint8).reshape(
                    (image.height, image.width, 3))
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif image.color_format == agibot_gdk.ColorFormat.BGR:
                return np.frombuffer(image.data, dtype=np.uint8).reshape(
                    (image.height, image.width, 3))
            elif image.color_format == agibot_gdk.ColorFormat.GRAY8:
                img = np.frombuffer(image.data, dtype=np.uint8).reshape(
                    (image.height, image.width))
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif image.color_format == agibot_gdk.ColorFormat.RS2_FORMAT_Z16:
                # 深度图：16位无符号整数，单位通常为毫米
                depth = np.frombuffer(image.data, dtype=np.uint16).reshape(
                    (image.height, image.width))
                return depth  # 返回原始深度数组，单位为毫米
            else:
                raise ValueError(f"不支持的颜色格式: {image.color_format}")
        else:
            raise ValueError(f"不支持的编码格式: {image.encoding}")
    
    def get_depth_at_pixel(self, depth_img, x, y):
        """
        获取深度图中指定像素的距离（单位：米）
        假设深度图是16位，单位毫米，0表示无效
        """
        h, w = depth_img.shape
        if not (0 <= y < h and 0 <= x < w):
            return None
        
        depth_mm = depth_img[y, x]
        if depth_mm == 0:
            # 中心点无效，尝试周围3x3区域的中位数
            roi = depth_img[max(0, y-1):min(h, y+2), max(0, x-1):min(w, x+2)]
            valid = roi[roi > 0]
            if len(valid) > 0:
                depth_mm = np.median(valid)
            else:
                return None
        return depth_mm / 1000.0  # 转换为米
    
    def run(self):
        """主循环：获取图像、检测、保存结果"""
        print("开始实时处理，按 Ctrl+C 停止")
        frame_count = 0
        
        try:
            while True:
                # 获取彩色图和深度图
                color_sdk = self.camera.get_latest_image(self.color_camera_type, 1000.0)
                depth_sdk = self.camera.get_latest_image(self.depth_camera_type, 1000.0)
                
                if color_sdk is None or depth_sdk is None:
                    print("等待图像数据...")
                    time.sleep(0.1)
                    continue
                
                # 解码图像
                color_img = self.decode_image(color_sdk)
                depth_img = self.decode_image(depth_sdk)  # 返回numpy数组 (h, w)
                
                # YOLO检测
                results = self.model(color_img, verbose=False)[0]
                
                # 在图像上绘制检测结果并获取距离
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = results.names[cls_id]
                    
                    #只处理类别为'bottle'的目标
                    if label != 'bottle':
                        continue
                    # 计算中心点
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    
                    # 获取距离
                    distance = self.get_depth_at_pixel(depth_img, cx, cy)
                    if distance is not None:
                        info = f"{label} {distance:.2f}m"
                    else:
                        info = f"{label} (无深度)"
                    
                    # 绘制边界框和标签
                    cv2.rectangle(color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(color_img, info, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # 打印距离信息到终端
                    print(f"帧 {frame_count:06d}: {info}")
                
                # 每10帧保存一次图像（避免文件过多）
                if frame_count % 10 == 0:
                    filename = f"output_frames/frame_{frame_count:06d}.jpg"
                    cv2.imwrite(filename, color_img)
                    print(f"已保存图像: {filename}")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n用户中断程序")
        finally:
            # 关闭相机
            try:
                self.camera.close_camera()
                print("相机已关闭")
            except:
                pass
            print("程序结束")

def main():
    demo = YOLOWithDistance()
    demo.run()

if __name__ == "__main__":
    main()