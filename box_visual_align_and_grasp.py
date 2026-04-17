#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
box_visual_align_and_grasp.py

流程：
1. 到拍照位：view_waist + view_pose_calib
2. 右手相机 YOLO 检测箱子
3. 用拍照位模板误差 + 局部 Jacobian 逆矩阵，算出修正量 dx/dy/dz
4. 到抓取位：grasp_waist + grasp_pose
5. 在抓取位基础上叠加同样的 dx/dy/dz
6. 可选闭夹爪抓取

建议第一次先 dry-run（不闭夹爪）跑通。
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from g2_robot_controller import G2RobotController


class BoxVisualAlignAndGrasp:
    def __init__(
        self,
        model_path: str,
        class_name: str = "box",
        camera_name: str = "right",
        conf: float = 0.5,
        retries: int = 8,
        timeout_ms: float = 1000.0,
        save_dir: str = "align_debug_yolo",
        gain: float = 0.5,
        max_dx: float = 0.015,
        max_dy: float = 0.015,
        max_dz: float = 0.010,
        do_grasp: bool = False,
        close_position: float = 1.0,
    ):
        self.model_path = model_path
        self.class_name = class_name
        self.camera_name = camera_name.lower().strip()
        self.conf = conf
        self.retries = retries
        self.timeout_ms = timeout_ms
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)z









        self.gain = gain
        self.max_dx = max_dx
        self.max_dy = max_dy
        self.max_dz = max_dz
        self.do_grasp = do_grasp
        self.close_position = close_position

        # 你当前测出来的拍照位模板值
        self.TARGET_CX = 640.0
        self.TARGET_CY = 410.0
        self.TARGET_H = 208.0

        # 用你当前三次实验拟合出来的局部逆 Jacobian
        # 输入误差 e = [cx - TARGET_CX, cy - TARGET_CY, h - TARGET_H]
        # 输出 [dx, dy, dz] (m)
        self.J_INV = np.array([
            [-0.00087108,  0.00052265,  0.00282230],
            [ 0.00153310,  0.00108014,  0.00383275],
            [-0.00101045, -0.00139373, -0.00752613],
        ], dtype=float)

        self.ctrl: Optional[G2RobotController] = None
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
        model_file = Path(self.model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"找不到模型文件: {model_file}")

        print("🔧 初始化控制器...")
        self.ctrl = G2RobotController()
        self.ctrl.init()

        print("📷 初始化相机...")
        self.camera = agibot_gdk.Camera()
        time.sleep(2.0)

        self.model = YOLO(str(model_file))
        print(f"🧠 YOLO 模型已加载: {model_file}")

        self._print_intrinsic()

    def release(self):
        if self.ctrl is not None:
            self.ctrl.release()

    def _print_intrinsic(self):
        try:
            intr = self.camera.get_camera_intrinsic(self.camera_type)
            vals = list(intr.intrinsic)
            if len(vals) >= 4:
                fx, fy, cx, cy = vals[:4]
                print(f"📐 相机内参: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        except Exception as e:
            print(f"⚠️ 获取相机内参失败，但不影响继续运行: {e}")

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

    def detect_box_stable(self) -> Dict[str, Any]:
        dets: List[Dict[str, Any]] = []
        last_frame = None
        last_image = None

        print(f"🎯 开始稳定检测，camera={self.camera_name}, retries={self.retries}")
        for i in range(self.retries):
            frame, image_obj = self.get_latest_frame()
            if frame is None:
                print(f"  [{i+1}/{self.retries}] 未取到有效图像")
                time.sleep(0.15)
                continue

            det = self.detect_once(frame)
            if det is None:
                print(f"  [{i+1}/{self.retries}] 未检测到目标类: {self.class_name}")
                time.sleep(0.15)
                continue

            print(
                f"  [{i+1}/{self.retries}] "
                f"conf={det['conf']:.3f}, center={det['center_xy']}, size={det['size_wh']}"
            )
            det["_frame"] = frame.copy()
            det["_image_obj"] = image_obj
            dets.append(det)
            last_frame = frame
            last_image = image_obj
            time.sleep(0.05)

        if not dets:
            raise RuntimeError("多次尝试后仍未检测到目标")

        # 用中位数提高稳定性
        cxs = np.array([d["center_xy"][0] for d in dets], dtype=float)
        cys = np.array([d["center_xy"][1] for d in dets], dtype=float)
        ws = np.array([d["size_wh"][0] for d in dets], dtype=float)
        hs = np.array([d["size_wh"][1] for d in dets], dtype=float)
        confs = np.array([d["conf"] for d in dets], dtype=float)

        cx = int(round(np.median(cxs)))
        cy = int(round(np.median(cys)))
        w = int(round(np.median(ws)))
        h = int(round(np.median(hs)))

        # 选一个最接近中位数中心的框用于画图
        best_idx = int(np.argmin((cxs - cx) ** 2 + (cys - cy) ** 2))
        chosen = dets[best_idx]

        x1, y1, x2, y2 = chosen["bbox_xyxy"]
        frame = chosen["_frame"]
        image_obj = chosen["_image_obj"]

        return {
            "class_name": self.class_name,
            "conf": float(np.max(confs)),agi@G2:~/app/gdk/examples/python$ grep -n "move_right_arm" /home/agi/app/gdk/examples/python/g2_robot_controller.py
grep -n "move_left_arm" /home/agi/app/gdk/examples/python/g2_robot_controller.py
836:    def move_right_arm_to_point(
853:    def move_right_arm_servo(
873:    def move_right_arm_delta(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0, duration_s: float = 2.0) -> bool:
877:        return self.move_right_arm_servo(
894:    def move_right_arm_ee_to_point(self, name: str, duration_s: float = 2.0) -> bool:
901:        return self.move_right_arm_servo(data[:3], data[3:], duration_s=duration_s)
915:        return self._run_seq(names, self.move_right_arm_to_point, velocity, dwell_s, "右臂")
744:    def move_left_arm_to_point(
761:    def move_left_arm_servo(
781:    def move_left_arm_delta(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0, duration_s: float = 2.0) -> bool:
785:        return self.move_left_arm_servo(
802:    def move_left_arm_ee_to_point(self, name: str, duration_s: float = 2.0) -> bool:
809:        return self.move_left_arm_servo(data[:3], data[3:], duration_s=duration_s)
823:        return self._run_seq(names, self.move_left_arm_to_point, velocity, dwell_s, "左臂")
            agi@G2:~/app/gdk/examples/python$ 

            "bbox_xyxy": [x1, y1, x2, y2],
            "center_xy": [cx, cy],
            "size_wh": [w, h],
            "_frame": frame,
            "_image_obj": image_obj,
        }

    def compute_visual_delta(self, cx: float, cy: float, h: float) -> Tuple[float, float, float]:
        e = np.array([
            cx - self.TARGET_CX,
            cy - self.TARGET_CY,
            h  - self.TARGET_H,
        ], dtype=float)

        d = -self.gain * (self.J_INV @ e)

        dx = float(np.clip(d[0], -self.max_dx, self.max_dx))
        dy = float(np.clip(d[1], -self.max_dy, self.max_dy))
        dz = float(np.clip(d[2], -self.max_dz, self.max_dz))
        return dx, dy, dz

    def draw_detection(self, frame: np.ndarray, det: Dict[str, Any], dx: float, dy: float, dz: float) -> np.ndarray:
        img = frame.copy()
        x1, y1, x2, y2 = det["bbox_xyxy"]
        cx, cy = det["center_xy"]
        w, h = det["size_wh"]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
        cv2.circle(img, (int(self.TARGET_CX), int(self.TARGET_CY)), 5, (255, 255, 0), -1)

        cv2.putText(
            img,
            f"curr=({cx},{cy},{h}) target=({int(self.TARGET_CX)},{int(self.TARGET_CY)},{int(self.TARGET_H)})",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            img,
            f"delta: dx={dx:+.4f} dy={dy:+.4f} dz={dz:+.4f}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 255),
            2,
        )
        return img

    def save_debug_image(self, image_obj, img: np.ndarray) -> Path:
        ts = getattr(image_obj, "timestamp_ns", None)
        if ts is None:
            ts = int(time.time() * 1e9)
        out = self.save_dir / f"align_result_{self.camera_name}_{ts}.jpg"
        cv2.imwrite(str(out), img)
        return out

    def run(self):
        assert self.ctrl is not None

        print("\n========== 第 1 步：到拍照位 ==========")
        # self.ctrl.open_gripper()
        # time.sleep(0.8)
        self.ctrl.move_waist_to_point("view_waist")
        time.sleep(0.8)
        self.ctrl.move_both_arms_ee_to_point("view_pose_calib", duration_s=2.0)
        time.sleep(1.0)

        print("\n========== 第 2 步：拍照检测 ==========")
        det = self.detect_box_stable()
        cx, cy = det["center_xy"]
        _, h = det["size_wh"]

        print("\n========== 第 3 步：计算修正量 ==========")
        dx, dy, dz = self.compute_visual_delta(cx=cx, cy=cy, h=h)
        print(f"当前检测值: center=({cx},{cy}) h={h}")
        print(f"模板目标值: center=({int(self.TARGET_CX)},{int(self.TARGET_CY)}) h={int(self.TARGET_H)}")
        print(f"计算修正量: dx={dx:+.4f} m, dy={dy:+.4f} m, dz={dz:+.4f} m")

        debug_img = self.draw_detection(det["_frame"], det, dx, dy, dz)
        debug_path = self.save_debug_image(det["_image_obj"], debug_img)
        print(f"调试图已保存: {debug_path}")

        print("\n========== 第 4 步：到抓取位并叠加修正 ==========")
        self.ctrl.move_waist_to_point("grasp_waist")
        time.sleep(0.8)
        self.ctrl.move_both_arms_ee_to_point("grasp_pose", duration_s=2.0)
        time.sleep(1.0)

        self.ctrl.move_both_arms_delta(dx=dx, dy=dy, dz=dz, duration_s=1.0)
        time.sleep(1.0)

        if self.do_grasp:
            print("\n========== 第 5 步：闭夹爪 ==========")
            self.ctrl.close_gripper(position=self.close_position)
            time.sleep(1.0)
            print("✅ 已执行抓取")
        else:
            print("\n⚠️ 当前为 dry-run，只走到了修正后的抓取位，没有闭夹爪。")
            print("   确认位置无误后，加 --do-grasp 再执行真正抓取。")


def build_argparser():
    ap = argparse.ArgumentParser(description="拍照位 YOLO 对齐 -> 修正抓取位 -> 抓取")
    ap.add_argument("--model", required=True, help="本地 YOLO 模型路径，例如 /home/agi/app/gdk/examples/python/best.pt")
    ap.add_argument("--class-name", default="box", help="目标类别名，默认 box")
    ap.add_argument("--camera", default="right", choices=["left", "right"], help="使用左手或右手彩色相机")
    ap.add_argument("--conf", type=float, default=0.5, help="YOLO 检测置信度阈值")
    ap.add_argument("--retries", type=int, default=8, help="检测采样帧数，默认 8")
    ap.add_argument("--timeout-ms", type=float, default=1000.0, help="单次取图超时，默认 1000ms")
    ap.add_argument("--save-dir", default="align_debug_yolo", help="调试图片保存目录")
    ap.add_argument("--gain", type=float, default=0.5, help="视觉修正增益，默认 0.5")
    ap.add_argument("--max-dx", type=float, default=0.015, help="dx 最大限幅，默认 0.015m")
    ap.add_argument("--max-dy", type=float, default=0.015, help="dy 最大限幅，默认 0.015m")
    ap.add_argument("--max-dz", type=float, default=0.010, help="dz 最大限幅，默认 0.010m")
    ap.add_argument("--do-grasp", action="store_true", help="加上后会真正闭夹爪抓取")
    ap.add_argument("--close-position", type=float, default=1.0, help="闭夹爪位置，默认 1.0")
    return ap


def main():
    args = build_argparser().parse_args()

    app = BoxVisualAlignAndGrasp(
        model_path=args.model,
        class_name=args.class_name,
        camera_name=args.camera,
        conf=args.conf,
        retries=args.retries,
        timeout_ms=args.timeout_ms,
        save_dir=args.save_dir,
        gain=args.gain,
        max_dx=args.max_dx,
        max_dy=args.max_dy,
        max_dz=args.max_dz,
        do_grasp=args.do_grasp,
        close_position=args.close_position,
    )

    try:
        app.init()
        app.run()
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        raise
    finally:
        app.release()


if __name__ == "__main__":
    main()