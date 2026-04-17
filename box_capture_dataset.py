#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
box_capture_dataset.py

用途：
1. 初始化 GDK 和 Camera
2. 从左手或右手彩色相机连续采图
3. 将图片按时间戳保存到指定目录
4. 用于后续训练自定义 YOLO 箱子检测模型

示例：
python3 box_capture_dataset.py --camera left --save-dir dataset/images_raw/left --count 200 --interval 0.6
python3 box_capture_dataset.py --camera right --save-dir dataset/images_raw/right --count 150 --interval 0.8
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    print("❌ 缺少 opencv-python，请先安装：pip install opencv-python")
    sys.exit(1)

try:
    import agibot_gdk
except ImportError:
    print("❌ 未找到 agibot_gdk，请确认已在机器人环境中运行")
    sys.exit(1)


class BoxCaptureDataset:
    def __init__(self, camera_name: str, save_dir: str, count: int, interval: float, timeout_ms: float):
        self.camera_name = camera_name.lower().strip()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.count = count
        self.interval = interval
        self.timeout_ms = timeout_ms

        self.camera = None
        self.camera_type = self._resolve_camera_type(self.camera_name)

    def _resolve_camera_type(self, camera_name: str):
        if camera_name == "left":
            return agibot_gdk.CameraType.kHandLeftColor
        if camera_name == "right":
            return agibot_gdk.CameraType.kHandRightColor
        raise ValueError("camera 只能是 left 或 right")

    def init(self):
        res = agibot_gdk.gdk_init()
        if res != agibot_gdk.GDKRes.kSuccess:
            raise RuntimeError(f"GDK 初始化失败: {res}")
        print("🔧 GDK 初始化完成")

        self.camera = agibot_gdk.Camera()
        time.sleep(2.0)
        print("📷 Camera 初始化完成")

        try:
            shape = self.camera.get_image_shape(self.camera_type)
            print(f"📐 图像尺寸: {shape[0]} x {shape[1]}")
        except Exception as e:
            print(f"⚠️ 获取图像尺寸失败: {e}")

    def release(self):
        try:
            agibot_gdk.gdk_release()
            print("✅ GDK 已释放")
        except Exception as e:
            print(f"⚠️ GDK 释放异常: {e}")

    def decode_image(self, image) -> Optional[np.ndarray]:
        if image is None or not hasattr(image, "data") or image.data is None or len(image.data) == 0:
            return None

        try:
            if image.encoding == agibot_gdk.Encoding.JPEG:
                arr = np.frombuffer(image.data, np.uint8)
                return cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if image.encoding == agibot_gdk.Encoding.PNG:
                arr = np.frombuffer(image.data, np.uint8)
                return cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if image.encoding == agibot_gdk.Encoding.UNCOMPRESSED:
                if image.color_format == agibot_gdk.ColorFormat.RGB:
                    arr = np.frombuffer(image.data, dtype=np.uint8).reshape((image.height, image.width, 3))
                    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                if image.color_format == agibot_gdk.ColorFormat.BGR:
                    return np.frombuffer(image.data, dtype=np.uint8).reshape((image.height, image.width, 3))
                if image.color_format == agibot_gdk.ColorFormat.GRAY8:
                    arr = np.frombuffer(image.data, dtype=np.uint8).reshape((image.height, image.width))
                    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

            return None
        except Exception as e:
            print(f"⚠️ 图像解码失败: {e}")
            return None

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], Optional[object]]:
        try:
            image = self.camera.get_latest_image(self.camera_type, self.timeout_ms)
        except Exception as e:
            print(f"⚠️ 取图失败: {e}")
            return None, None
        frame = self.decode_image(image)
        return frame, image

    def save_frame(self, frame: np.ndarray, image_obj, idx: int) -> Path:
        ts = getattr(image_obj, "timestamp_ns", None)
        if ts is None:
            ts = int(time.time() * 1e9)
        out = self.save_dir / f"{self.camera_name}_{idx:04d}_{ts}.jpg"
        cv2.imwrite(str(out), frame)
        return out

    def run(self) -> int:
        ok = 0
        print(f"🎯 开始采图: camera={self.camera_name}, count={self.count}, interval={self.interval}s")
        for i in range(1, self.count + 1):
            frame, image_obj = self.get_latest_frame()
            if frame is None:
                print(f"  [{i}/{self.count}] 未取到有效图像，跳过")
                time.sleep(self.interval)
                continue
            path = self.save_frame(frame, image_obj, i)
            ok += 1
            print(f"  [{i}/{self.count}] 已保存: {path}")
            time.sleep(self.interval)

        print(f"\n✅ 采图结束，成功保存 {ok}/{self.count} 张")
        return 0 if ok > 0 else 2


def build_argparser():
    ap = argparse.ArgumentParser(description="G2 手部相机数据采集脚本")
    ap.add_argument("--camera", default="left", choices=["left", "right"], help="左手或右手彩色相机")
    ap.add_argument("--save-dir", required=True, help="图片保存目录")
    ap.add_argument("--count", type=int, default=200, help="保存图片张数")
    ap.add_argument("--interval", type=float, default=0.6, help="每张图之间的间隔秒数")
    ap.add_argument("--timeout-ms", type=float, default=1000.0, help="单次取图超时")
    return ap


def main():
    args = build_argparser().parse_args()
    app = BoxCaptureDataset(
        camera_name=args.camera,
        save_dir=args.save_dir,
        count=args.count,
        interval=args.interval,
        timeout_ms=args.timeout_ms,
    )
    try:
        app.init()
        code = app.run()
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
        code = 130
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        code = 1
    finally:
        app.release()
    sys.exit(code)


if __name__ == "__main__":
    main()