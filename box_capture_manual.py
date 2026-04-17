#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
box_capture_manual.py

手动采图脚本：
- 使用左手或右手彩色相机实时预览
- 你摆好一个位置后，按 s 保存一张
- 按 q 退出

示例：
python3 box_capture_manual.py --camera right --save-dir /home/agi/app/gdk/examples/python/box_dataset/images_raw/right
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


class ManualBoxCapture:
    def __init__(self, camera_name: str, save_dir: str, timeout_ms: float = 1000.0):
        self.camera_name = camera_name.lower().strip()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_ms = timeout_ms

        self.camera = None
        self.save_index = 0
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

    def save_frame(self, frame: np.ndarray, image_obj):
        self.save_index += 1
        ts = getattr(image_obj, "timestamp_ns", None)
        if ts is None:
            ts = int(time.time() * 1e9)

        out = self.save_dir / f"{self.camera_name}_{self.save_index:04d}_{ts}.jpg"
        cv2.imwrite(str(out), frame)
        print(f"💾 已保存: {out}")

    def run(self):
        print("\n操作说明：")
        print("  s : 保存当前帧")
        print("  q : 退出")
        print("  先摆好一个位置，再按 s 保存一张\n")

        while True:
            frame, image_obj = self.get_latest_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            show = frame.copy()
            cv2.putText(
                show,
                f"camera={self.camera_name}  saved={self.save_index}  [s]=save [q]=quit",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
            )

            cv2.imshow("box_capture_manual", show)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("s"):
                self.save_frame(frame, image_obj)
                time.sleep(0.15)
            elif key == ord("q"):
                break

        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="G2 手动采图脚本")
    ap.add_argument("--camera", default="right", choices=["left", "right"], help="左右手彩色相机")
    ap.add_argument("--save-dir", required=True, help="图片保存目录")
    ap.add_argument("--timeout-ms", type=float, default=1000.0, help="单次取图超时")
    args = ap.parse_args()

    app = ManualBoxCapture(
        camera_name=args.camera,
        save_dir=args.save_dir,
        timeout_ms=args.timeout_ms,
    )

    try:
        app.init()
        app.run()
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    finally:
        app.release()


if __name__ == "__main__":
    main()