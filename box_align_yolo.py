#!/usr/bin/env python3
"""
box_align_yolo.py

功能：
  1. 使用 YOLO 模型检测箱子
  2. 根据偏差自动调整底盘位置
  3. 循环直到箱子在图像中心并旋转对齐
"""

import time, math, os
import cv2
import numpy as np
import agibot_gdk
from ultralytics import YOLO

# ── 参数配置 ─────────────────────────
HEAD_CAM    = agibot_gdk.CameraType.kHeadColor
CAM_TIMEOUT = 1000.0
IMG_W, IMG_H = 640, 480

CENTER_TOL_X   = 30    # 水平偏差容忍（像素）
CENTER_TOL_ANG = 5     # 旋转角度容忍（度）
CHASSIS_LINEAR_Y  = 0.08   # 左右蟹行速度 m/s
CHASSIS_ANGULAR_Z = 0.10   # 旋转速度 rad/s
CTRL_DT           = 0.15   # 控制周期 s
MAX_STEPS         = 60    # 最大修正次数

YOLO_MODEL_PATH = "/home/agi/app/gdk/examples/python/best.pt"

# ── 底盘控制 ─────────────────────────
def send_chassis(pnc, vy, wz, dt):
    twist = agibot_gdk.Twist()
    twist.linear  = agibot_gdk.Vector3()
    twist.angular = agibot_gdk.Vector3()
    twist.linear.x  = 0.0
    twist.linear.y  = vy
    twist.angular.z = wz
    pnc.move_chassis(twist)
    time.sleep(dt)

def stop_chassis(pnc):
    twist = agibot_gdk.Twist()
    twist.linear  = agibot_gdk.Vector3()
    twist.angular = agibot_gdk.Vector3()
    pnc.move_chassis(twist)

# ── YOLO检测箱子 ─────────────────────
def detect_box(frame, model):
    results = model(frame)
    if not results or len(results[0].boxes) == 0:
        return None
    # 取第一个检测框
    box = results[0].boxes.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = box
    cx = (x1 + x2)/2
    cy = (y1 + y2)/2
    w = x2 - x1
    h = y2 - y1
    angle = 0.0  # 如果想更精确，可结合 minAreaRect
    return cx, cy, angle, int(w*h), [int(x1), int(y1), int(x2), int(y2)]

# ── 主循环 ─────────────────────────
def main():
    print("="*50)
    print("  YOLO 箱子对中程序")
    print("="*50)

    agibot_gdk.gdk_init()
    camera = agibot_gdk.Camera()
    pnc    = agibot_gdk.Pnc()
    time.sleep(2)

    # 请求底盘控制
    pnc.request_chassis_control(1)
    time.sleep(0.5)

    # 加载 YOLO 模型
    print("[YOLO] 加载模型...")
    model = YOLO(YOLO_MODEL_PATH)
    print("[YOLO] ✅ 模型就绪")

    os.makedirs("align_debug_yolo", exist_ok=True)

    for step in range(MAX_STEPS):
        # 拍图
        img = camera.get_latest_image(HEAD_CAM, CAM_TIMEOUT)
        if img is None:
            print(f"[{step:02d}] ⚠️ 拍图失败")
            time.sleep(0.3)
            continue

        # 转成BGR
        arr = np.frombuffer(bytes(img.data), dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        cx_img = frame.shape[1] // 2

        # 检测箱子
        res = detect_box(frame, model)
        if res is None:
            print(f"[{step:02d}] ❌ 未检测到箱子")
            cv2.imwrite(f"align_debug_yolo/step_{step:02d}_no_box.jpg", frame)
            time.sleep(0.3)
            continue

        cx, cy, angle, area, box_pts = res
        dx = cx - cx_img

        # debug 可视化
        cv2.rectangle(frame, (box_pts[0], box_pts[1]), (box_pts[2], box_pts[3]), (0,255,0),2)
        cv2.circle(frame, (int(cx), int(cy)), 5, (0,0,255), -1)
        cv2.line(frame, (cx_img,0), (cx_img,frame.shape[0]), (255,0,0), 1)
        cv2.imwrite(f"align_debug_yolo/step_{step:02d}.jpg", frame)

        print(f"[{step:02d}] dx={dx:+.0f}px angle={angle:+.1f}° area={area}")

        aligned_x = abs(dx) < CENTER_TOL_X
        aligned_ang = abs(angle) < CENTER_TOL_ANG

        if aligned_x and aligned_ang:
            print(f"✅ 对中完成！偏差 dx={dx:+.0f}px angle={angle:+.1f}°")
            break

        # 计算控制
        vy = -math.copysign(CHASSIS_LINEAR_Y, dx) if not aligned_x else 0.0
        wz = -math.copysign(CHASSIS_ANGULAR_Z, angle) if not aligned_ang else 0.0

        send_chassis(pnc, vy, wz, CTRL_DT)
    else:
        print("⚠️ 达到最大步数停止")

    stop_chassis(pnc)
    try:
        task = pnc.get_task_state()
        pnc.cancel_task(task.id)
    except Exception:
        pass

    try:
        camera.close_camera()
    except Exception:
        pass

    agibot_gdk.gdk_release()
    print("程序结束")

if __name__ == "__main__":
    main()