#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
box_detect_test.py

用途：
1. 初始化 GDK 与 Camera
2. 从左手/右手彩色相机取图
3. 用本地 YOLO 模型检测箱子
4. 保存原图和带框结果图
5. 打印 bbox / 中心点 / 置信度，给后续视觉纠偏用

示例：
python3 box_detect_test.py --model /home/agi/app/gdk/examples/python/box_yolo.pt --class-name box --camera left
python3 box_detect_test.py --model /home/agi/app/gdk/examples/python/box_yolo.pt --class-name box --camera right --conf 0.35 --retries 10
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np

try:
    import cv2
except ImportError:
    print("❌ 缺少 opencv-python，请先安装：pip install opencv-python")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ 缺少 ultralytics，请先安装：pip install ultralytics")
    sys.exit(1)

try:
    import agibot_gdk
except ImportError:
    print("❌ 未找到 agibot_gdk，请确认已在机器人环境中运行")
    sys.exit(1)


class BoxDetectTest:
    def __init__(
        self,
        model_path: str,
        class_name: str = "box",
        camera_name: str = "left",
        conf: float = 0.4,
        retries: int = 8,
        timeout_ms: float = 1000.0,
        save_dir: str = "vision_debug",
    ):
        self.model_path = model_path
        self.class_name = class_name
        self.camera_name = camera_name.lower().strip()
        self.conf = conf
        self.retries = retries
        self.timeout_ms = timeout_ms
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.camera = None
        self.model = None
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

        model_file = Path(self.model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"找不到模型文件: {model_file}")
        self.model = YOLO(str(model_file))
        print(f"🧠 YOLO 模型已加载: {model_file}")

        self._print_intrinsic()

    def release(self):
        try:
            agibot_gdk.gdk_release()
            print("✅ GDK 已释放")
        except Exception as e:
            print(f"⚠️ GDK 释放异常: {e}")

    def _print_intrinsic(self):
        try:
            intr = self.camera.get_camera_intrinsic(self.camera_type)
            vals = list(intr.intrinsic)
            if len(vals) >= 4:
                fx, fy, cx, cy = vals[:4]
                print(f"📐 相机内参: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
            else:
                print("⚠️ 相机内参返回长度不足，跳过打印")
        except Exception as e:
            print(f"⚠️ 获取相机内参失败，但不影响本脚本继续测试: {e}")

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

            print(f"⚠️ 不支持的图像格式: encoding={image.encoding}, color_format={image.color_format}")
            return None
        except Exception as e:
            print(f"⚠️ 图像解码失败: {e}")
            return None

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], Optional[Any]]:
        try:
            image = self.camera.get_latest_image(self.camera_type, self.timeout_ms)
        except Exception as e:
            print(f"⚠️ 取图失败: {e}")
            return None, None

        frame = self.decode_image(image)
        return frame, image

    def detect_once(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        results = self.model(frame, conf=self.conf, verbose=False)
        if not results:
            return None

        best = None
        best_conf = -1.0

        for r in results:
            names = r.names
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = names[cls_id]
                conf = float(box.conf[0])
                if cls_name != self.class_name:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                w = int(x2 - x1)
                h = int(y2 - y1)

                if conf > best_conf:
                    best_conf = conf
                    best = {
                        "class_name": cls_name,
                        "conf": conf,
                        "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                        "center_xy": [cx, cy],
                        "size_wh": [w, h],
                    }

        return best

    def draw_detection(self, frame: np.ndarray, det: Dict[str, Any]) -> np.ndarray:
        img = frame.copy()
        x1, y1, x2, y2 = det["bbox_xyxy"]
        cx, cy = det["center_xy"]
        w, h = det["size_wh"]
        conf = det["conf"]
        cls_name = det["class_name"]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
        label = f"{cls_name} {conf:.3f}  cx={cx} cy={cy} w={w} h={h}"
        cv2.putText(img, label, (max(10, x1), max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        return img

    def save_debug_images(self, frame: np.ndarray, annotated: np.ndarray, image_obj=None):
        ts = getattr(image_obj, "timestamp_ns", None)
        if ts is None:
            ts = int(time.time() * 1e9)

        raw_path = self.save_dir / f"raw_{self.camera_name}_{ts}.jpg"
        ann_path = self.save_dir / f"det_{self.camera_name}_{ts}.jpg"

        cv2.imwrite(str(raw_path), frame)
        cv2.imwrite(str(ann_path), annotated)
        return raw_path, ann_path

    def run(self) -> int:
        best_det = None
        best_frame = None
        best_image_obj = None

        print(f"🎯 开始检测，camera={self.camera_name}, class={self.class_name}, retries={self.retries}")

        for i in range(self.retries):
            frame, image_obj = self.get_latest_frame()
            if frame is None:
                print(f"  [{i+1}/{self.retries}] 未取到有效图像")
                time.sleep(0.2)
                continue

            det = self.detect_once(frame)
            if det is None:
                print(f"  [{i+1}/{self.retries}] 未检测到目标类: {self.class_name}")
                time.sleep(0.2)
                continue

            print(
                f"  [{i+1}/{self.retries}] 检测到 {det['class_name']}: "
                f"conf={det['conf']:.3f}, center={det['center_xy']}, size={det['size_wh']}"
            )

            if best_det is None or det["conf"] > best_det["conf"]:
                best_det = det
                best_frame = frame.copy()
                best_image_obj = image_obj

        if best_det is None:
            print("❌ 多次尝试后仍未检测到目标")
            return 2

        annotated = self.draw_detection(best_frame, best_det)
        raw_path, ann_path = self.save_debug_images(best_frame, annotated, best_image_obj)

        print("\n✅ 检测成功")
        print(f"   class_name : {best_det['class_name']}")
        print(f"   conf       : {best_det['conf']:.4f}")
        print(f"   bbox_xyxy  : {best_det['bbox_xyxy']}")
        print(f"   center_xy  : {best_det['center_xy']}")
        print(f"   size_wh    : {best_det['size_wh']}")
        print(f"   raw_image  : {raw_path}")
        print(f"   det_image  : {ann_path}")
        print("\n下一步就可以在这个 bbox ROI 里做箱口矩形精提。")
        return 0


def build_argparser():
    ap = argparse.ArgumentParser(description="G2 手部相机 + YOLO 箱子检测测试脚本")
    ap.add_argument("--model", required=True, help="本地 YOLO 模型路径，例如 /home/agi/app/gdk/examples/python/box_yolo.pt")
    ap.add_argument("--class-name", default="box", help="目标类别名，默认 box")
    ap.add_argument("--camera", default="left", choices=["left", "right"], help="使用左手或右手彩色相机")
    ap.add_argument("--conf", type=float, default=0.4, help="YOLO 检测置信度阈值")
    ap.add_argument("--retries", type=int, default=8, help="最多尝试多少帧，默认 8")
    ap.add_argument("--timeout-ms", type=float, default=1000.0, help="单次取图超时，默认 1000ms")
    ap.add_argument("--save-dir", default="vision_debug", help="调试图片保存目录")
    return ap


def main():
    args = build_argparser().parse_args()

    tester = BoxDetectTest(
        model_path=args.model,
        class_name=args.class_name,
        camera_name=args.camera,
        conf=args.conf,
        retries=args.retries,
        timeout_ms=args.timeout_ms,
        save_dir=args.save_dir,
    )

    try:
        tester.init()
        code = tester.run()
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
        code = 130
    except Exception as e:
        print(f"❌ 脚本运行失败: {e}")
        code = 1
    finally:
        tester.release()

    sys.exit(code)


if __name__ == "__main__":
    main()