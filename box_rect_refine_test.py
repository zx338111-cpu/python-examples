#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
box_rect_refine_test.py

用途：
1. 初始化 GDK + Camera
2. 用本地 YOLO 模型检测箱子
3. 在 YOLO bbox ROI 内用 OpenCV 做矩形精提
4. 输出矩形中心、宽高、角度、四角点
5. 保存调试图，给后续视觉纠偏使用
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


class BoxRectRefineTest:
    def __init__(
        self,
        model_path: str,
        class_name: str = "box",
        camera_name: str = "right",
        conf: float = 0.5,
        retries: int = 8,
        timeout_ms: float = 1000.0,
        save_dir: str = "vision_debug",
        roi_pad: int = 12,
    ):
        self.model_path = model_path
        self.class_name = class_name
        self.camera_name = camera_name.lower().strip()
        self.conf = conf
        self.retries = retries
        self.timeout_ms = timeout_ms
        self.roi_pad = roi_pad
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

    def detect_box(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
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

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                w = int(x2 - x1)
                h = int(y2 - y1)

                if conf > best_conf:
                    best_conf = conf
                    best = {
                        "class_name": cls_name,
                        "conf": conf,
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "center_xy": [cx, cy],
                        "size_wh": [w, h],
                    }
        return best

    def _normalize_angle(self, rect_w: float, rect_h: float, angle_deg: float) -> float:
        if rect_w < rect_h:
            angle_deg += 90.0
        while angle_deg >= 90.0:
            angle_deg -= 180.0
        while angle_deg < -90.0:
            angle_deg += 180.0
        return angle_deg

    def refine_rect(self, frame: np.ndarray, det: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        h_img, w_img = frame.shape[:2]
        x1, y1, x2, y2 = det["bbox_xyxy"]

        rx1 = clamp(x1 - self.roi_pad, 0, w_img - 1)
        ry1 = clamp(y1 - self.roi_pad, 0, h_img - 1)
        rx2 = clamp(x2 + self.roi_pad, 0, w_img - 1)
        ry2 = clamp(y2 + self.roi_pad, 0, h_img - 1)

        if rx2 <= rx1 or ry2 <= ry1:
            return None

        roi = frame[ry1:ry2, rx1:rx2].copy()
        roi_h, roi_w = roi.shape[:2]
        roi_area = float(roi_h * roi_w)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blur, 50, 150)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

        candidates = []
        for mode_name, bin_img in [("edges", edges), ("thresh", th)]:
            contours, _ = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < roi_area * 0.03 or area > roi_area * 0.95:
                    continue

                rect = cv2.minAreaRect(c)
                (cx, cy), (rw, rh), angle = rect
                if rw < 20 or rh < 20:
                    continue

                rect_area = max(rw * rh, 1.0)
                fill_ratio = float(area / rect_area)
                if fill_ratio < 0.25:
                    continue

                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.03 * peri, True)
                approx_n = len(approx)

                score = area * (0.6 + 0.4 * min(fill_ratio, 1.0))
                if 4 <= approx_n <= 6:
                    score *= 1.15

                candidates.append({
                    "mode": mode_name,
                    "area": area,
                    "fill_ratio": fill_ratio,
                    "approx_n": approx_n,
                    "rect": rect,
                    "score": score,
                    "edges": edges,
                    "thresh": th,
                    "roi": roi,
                    "roi_box": [rx1, ry1, rx2, ry2],
                })

        if not candidates:
            return None

        best = max(candidates, key=lambda x: x["score"])
        rect = best["rect"]
        (cx, cy), (rw, rh), angle = rect
        angle_norm = self._normalize_angle(rw, rh, angle)

        pts = cv2.boxPoints(rect)
        pts = np.int32(np.round(pts))

        pts_global = pts.copy()
        pts_global[:, 0] += rx1
        pts_global[:, 1] += ry1

        return {
            "roi_box_xyxy": [rx1, ry1, rx2, ry2],
            "roi_size_wh": [roi_w, roi_h],
            "rect_center_xy_local": [float(cx), float(cy)],
            "rect_center_xy_global": [float(cx + rx1), float(cy + ry1)],
            "rect_size_wh": [float(rw), float(rh)],
            "rect_angle_deg": float(angle_norm),
            "rect_points_local": pts.tolist(),
            "rect_points_global": pts_global.tolist(),
            "score": float(best["score"]),
            "area": float(best["area"]),
            "fill_ratio": float(best["fill_ratio"]),
            "approx_n": int(best["approx_n"]),
            "mode": best["mode"],
            "roi": best["roi"],
            "edges": best["edges"],
            "thresh": best["thresh"],
        }

    def draw_yolo(self, frame: np.ndarray, det: Dict[str, Any]) -> np.ndarray:
        img = frame.copy()
        x1, y1, x2, y2 = det["bbox_xyxy"]
        cx, cy = det["center_xy"]
        w, h = det["size_wh"]
        conf = det["conf"]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
        text = f"YOLO box conf={conf:.3f} center=({cx},{cy}) size=({w},{h})"
        cv2.putText(img, text, (max(10, x1), max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        return img

    def draw_refine(self, frame: np.ndarray, det: Dict[str, Any], ref: Dict[str, Any]) -> np.ndarray:
        img = self.draw_yolo(frame, det)
        rx1, ry1, rx2, ry2 = ref["roi_box_xyxy"]
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (255, 200, 0), 1)

        pts = np.array(ref["rect_points_global"], dtype=np.int32)
        cv2.polylines(img, [pts], True, (255, 0, 255), 2)

        cx, cy = ref["rect_center_xy_global"]
        cx_i, cy_i = int(round(cx)), int(round(cy))
        cv2.circle(img, (cx_i, cy_i), 5, (0, 255, 255), -1)

        rw, rh = ref["rect_size_wh"]
        angle = ref["rect_angle_deg"]
        text = f"Rect center=({cx_i},{cy_i}) size=({rw:.1f},{rh:.1f}) angle={angle:.1f}deg"
        cv2.putText(img, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        return img

    def save_debug_images(self, image_obj, frame, yolo_img, ref):
        ts = getattr(image_obj, "timestamp_ns", None)
        if ts is None:
            ts = int(time.time() * 1e9)

        raw_path = self.save_dir / f"raw_{self.camera_name}_{ts}.jpg"
        yolo_path = self.save_dir / f"yolo_{self.camera_name}_{ts}.jpg"
        roi_path = self.save_dir / f"roi_{self.camera_name}_{ts}.jpg"
        refine_path = self.save_dir / f"refine_{self.camera_name}_{ts}.jpg"
        edge_path = self.save_dir / f"edge_{self.camera_name}_{ts}.jpg"
        thresh_path = self.save_dir / f"thresh_{self.camera_name}_{ts}.jpg"

        cv2.imwrite(str(raw_path), frame)
        cv2.imwrite(str(yolo_path), yolo_img)
        cv2.imwrite(str(roi_path), ref["roi"])
        cv2.imwrite(str(refine_path), ref["refine_img"])
        cv2.imwrite(str(edge_path), ref["edges"])
        cv2.imwrite(str(thresh_path), ref["thresh"])

        return {
            "raw": raw_path,
            "yolo": yolo_path,
            "roi": roi_path,
            "refine": refine_path,
            "edge": edge_path,
            "thresh": thresh_path,
        }

    def run(self) -> int:
        best_pack = None

        print(f"🎯 开始检测+矩形精提，camera={self.camera_name}, class={self.class_name}, retries={self.retries}")
        for i in range(self.retries):
            frame, image_obj = self.get_latest_frame()
            if frame is None:
                print(f"  [{i+1}/{self.retries}] 未取到有效图像")
                time.sleep(0.2)
                continue

            det = self.detect_box(frame)
            if det is None:
                print(f"  [{i+1}/{self.retries}] 未检测到目标类: {self.class_name}")
                time.sleep(0.2)
                continue

            ref = self.refine_rect(frame, det)
            if ref is None:
                print(f"  [{i+1}/{self.retries}] YOLO 成功，但矩形精提失败: center={det['center_xy']} size={det['size_wh']}")
                time.sleep(0.2)
                continue

            cx, cy = ref["rect_center_xy_global"]
            rw, rh = ref["rect_size_wh"]
            angle = ref["rect_angle_deg"]
            print(
                f"  [{i+1}/{self.retries}] "
                f"box_conf={det['conf']:.3f}, rect_center=({cx:.1f},{cy:.1f}), "
                f"rect_size=({rw:.1f},{rh:.1f}), angle={angle:.1f}, mode={ref['mode']}, score={ref['score']:.1f}"
            )

            if best_pack is None or det["conf"] > best_pack["det"]["conf"]:
                best_pack = {"frame": frame.copy(), "image_obj": image_obj, "det": det, "ref": ref}

        if best_pack is None:
            print("❌ 多次尝试后仍未成功完成矩形精提")
            return 2

        frame = best_pack["frame"]
        image_obj = best_pack["image_obj"]
        det = best_pack["det"]
        ref = best_pack["ref"]

        yolo_img = self.draw_yolo(frame, det)
        refine_img = self.draw_refine(frame, det, ref)
        ref["refine_img"] = refine_img

        paths = self.save_debug_images(image_obj, frame, yolo_img, ref)

        print("\n✅ 检测+矩形精提成功")
        print(f"   YOLO bbox_xyxy         : {det['bbox_xyxy']}")
        print(f"   YOLO center_xy         : {det['center_xy']}")
        print(f"   YOLO size_wh           : {det['size_wh']}")
        print(f"   Rect center_xy_global  : {[round(v, 2) for v in ref['rect_center_xy_global']]}")
        print(f"   Rect size_wh           : {[round(v, 2) for v in ref['rect_size_wh']]}")
        print(f"   Rect angle_deg         : {round(ref['rect_angle_deg'], 2)}")
        print(f"   Rect points_global     : {ref['rect_points_global']}")
        print(f"   Refine mode            : {ref['mode']}")
        print(f"   Refine score           : {round(ref['score'], 2)}")
        print(f"   Saved raw              : {paths['raw']}")
        print(f"   Saved yolo             : {paths['yolo']}")
        print(f"   Saved roi              : {paths['roi']}")
        print(f"   Saved refine           : {paths['refine']}")
        print(f"   Saved edge             : {paths['edge']}")
        print(f"   Saved thresh           : {paths['thresh']}")
        print("\n下一步就可以把当前矩形结果和理想抓取模板做误差比较，再换算成机械臂小步修正量。")
        return 0


def build_argparser():
    ap = argparse.ArgumentParser(description="G2 右/左手相机 + YOLO + 矩形精提测试脚本")
    ap.add_argument("--model", required=True, help="本地 YOLO 模型路径，例如 /home/agi/app/gdk/examples/python/best.pt")
    ap.add_argument("--class-name", default="box", help="目标类别名，默认 box")
    ap.add_argument("--camera", default="right", choices=["left", "right"], help="使用左手或右手彩色相机")
    ap.add_argument("--conf", type=float, default=0.5, help="YOLO 检测置信度阈值")
    ap.add_argument("--retries", type=int, default=8, help="最多尝试多少帧，默认 8")
    ap.add_argument("--timeout-ms", type=float, default=1000.0, help="单次取图超时，默认 1000ms")
    ap.add_argument("--save-dir", default="vision_debug", help="调试图片保存目录")
    ap.add_argument("--roi-pad", type=int, default=12, help="YOLO bbox 向外扩的像素，默认 12")
    return ap


def main():
    args = build_argparser().parse_args()

    app = BoxRectRefineTest(
        model_path=args.model,
        class_name=args.class_name,
        camera_name=args.camera,
        conf=args.conf,
        retries=args.retries,
        timeout_ms=args.timeout_ms,
        save_dir=args.save_dir,
        roi_pad=args.roi_pad,
    )

    try:
        app.init()
        code = app.run()
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
        code = 130
    except Exception as e:
        print(f"❌ 脚本运行失败: {e}")
        code = 1
    finally:
        app.release()

    sys.exit(code)


if __name__ == "__main__":
    main()