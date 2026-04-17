#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
box_capture_manual_headless.py

适用于 SSH / 无桌面环境的手动采图脚本：
- 不使用 cv2.imshow，因此不会触发 xcb / DISPLAY 错误
- 持续读取右手或左手彩色相机最新图像
- 你在终端里按 s，就把“当前最新一帧”保存下来
- 按 q 退出

示例：
python3 box_capture_manual_headless.py \
  --camera right \
  --save-dir /home/agi/app/gdk/examples/python/box_dataset/images_raw/right
"""

import argparse
import sys
import time
import select
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

try:
    import tty
    import termios
except ImportError:
    print("❌ 当前环境不支持终端原始按键读取")
    sys.exit(1)


class ManualBoxCaptureHeadless:
    def __init__(self, camera_name: str, save_dir: str, timeout_ms: float = 1000.0):
        self.camera_name = camera_name.lower().strip()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_ms = timeout_ms

        self.camera = None
        self.save_index = 0
        self.camera_type = self._resolve_camera_type(self.camera_name)

        self.latest_frame = None
        self.latest_image_obj = None

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

    def save_frame(self):
        if self.latest_frame is None:
            print("⚠️ 当前还没有可保存的图像")
            return

        self.save_index += 1
        ts = getattr(self.latest_image_obj, "timestamp_ns", None)
        if ts is None:
            ts = int(time.time() * 1e9)

        out = self.save_dir / f"{self.camera_name}_{self.save_index:04d}_{ts}.jpg"
        cv2.imwrite(str(out), self.latest_frame)
        print(f"\n💾 已保存: {out}")

    def _read_key_nonblocking(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def run(self):
        print("\n操作说明：")
        print("  s : 保存当前最新一帧")
        print("  q : 退出")
        print("  这个版本不弹窗，适合 SSH / 无桌面环境")
        print("  你先摆好位置，再在终端按 s 保存\n")

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        last_status_time = 0.0

        try:
            tty.setcbreak(fd)

            while True:
                frame, image_obj = self.get_latest_frame()
                if frame is not None:
                    self.latest_frame = frame
                    self.latest_image_obj = image_obj

                now = time.time()
                if now - last_status_time > 1.0:
                    if image_obj is not None:
                        ts = getattr(image_obj, "timestamp_ns", 0)
                        print(f"\r📡 正在取图... saved={self.save_index} latest_ts={ts}", end="", flush=True)
                    else:
                        print(f"\r📡 正在取图... saved={self.save_index} latest_ts=None", end="", flush=True)
                    last_status_time = now

                key = self._read_key_nonblocking()
                if key == "s":
                    self.save_frame()
                elif key == "q":
                    print()
                    break

                time.sleep(0.03)

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main():
    ap = argparse.ArgumentParser(description="G2 无桌面环境手动采图脚本")
    ap.add_argument("--camera", default="right", choices=["left", "right"], help="左右手彩色相机")
    ap.add_argument("--save-dir", required=True, help="图片保存目录")
    ap.add_argument("--timeout-ms", type=float, default=1000.0, help="单次取图超时")
    args = ap.parse_args()

    app = ManualBoxCaptureHeadless(
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
