#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
g2_visual_perception.py — 智元 G2 机器人视觉感知中枢

类：G2VisualPerception
职责：
  1. 管理硬件（相机生命周期、重连）
  2. 加载与管理标定参数（内参、手眼外参）
  3. 加载与热更新 ONNX/YOLOv8 检测模型
  4. 执行检测推理，返回结构化结果
  5. 2D → 3D 坐标估计（深度图或双目）
  6. 坐标系变换（相机系 → 机械臂基坐标系）
  7. 手眼标定数据采集与计算
  8. 训练样本采集与离线训练触发
  9. 健康监控、日志、安全停止

与手臂类的交互接口：
  - get_execution_target(object_id) → PoseTarget
  - safe_stop() → 通知外部手臂类暂停

依赖：
  agibot_gdk, numpy, opencv-python, pyyaml
  可选: onnxruntime, ultralytics (YOLOv8)

作者：Claude / 2026-04
"""

from __future__ import annotations

import copy
import json
import logging
import os
import queue
import subprocess
import threading
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    warnings.warn("OpenCV 未安装，视觉功能受限。pip install opencv-python")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    warnings.warn("PyYAML 未安装。pip install pyyaml")

try:
    import agibot_gdk
    HAS_GDK = True
except ImportError:
    HAS_GDK = False
    warnings.warn("agibot_gdk 未找到，进入 Mock 模式（仅供开发调试）")

# ──────────────────────────────────────────────────────────────
# 日志配置
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][G2VP] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("G2VisualPerception")


# ══════════════════════════════════════════════════════════════
# 数据结构定义
# ══════════════════════════════════════════════════════════════

class CameraID(Enum):
    """G2 支持的相机枚举（对齐 agibot_gdk.CameraType）"""
    HEAD_STEREO_LEFT  = "kHeadStereoLeft"
    HEAD_STEREO_RIGHT = "kHeadStereoRight"
    HEAD_COLOR        = "kHeadColor"
    HEAD_DEPTH        = "kHeadDepth"
    HAND_LEFT         = "kHandLeftColor"
    HAND_RIGHT        = "kHandRightColor"


@dataclass
class CameraIntrinsics:
    """相机内参"""
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    width: int = 0
    height: int = 0
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5))

    @property
    def K(self) -> np.ndarray:
        """3×3 内参矩阵"""
        return np.array([
            [self.fx,    0.0, self.cx],
            [   0.0, self.fy, self.cy],
            [   0.0,    0.0,    1.0],
        ], dtype=np.float64)


@dataclass
class CalibrationData:
    """标定数据容器"""
    # 各相机内参
    intrinsics: Dict[str, CameraIntrinsics] = field(default_factory=dict)
    # 手眼变换矩阵：相机坐标系 → 机械臂基坐标系 (4×4 齐次矩阵)
    T_cam2base: Optional[np.ndarray] = None
    # 相机坐标系 → 末端执行器坐标系 (eye-in-hand 标定时使用)
    T_cam2tool: Optional[np.ndarray] = None
    # 双目基线（m）
    stereo_baseline: float = 0.065
    # 标定文件路径（加载来源，用于热更新判断）
    source_path: str = ""
    load_time: float = 0.0


@dataclass
class Detection2D:
    """单个 2D 检测结果"""
    object_id: str          # 类别标签
    track_id: int = -1      # 跟踪 ID（-1 表示无跟踪）
    confidence: float = 0.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)   # (x1, y1, x2, y2) 像素
    mask: Optional[np.ndarray] = None                  # 分割掩码（可选）
    timestamp_ns: int = 0


@dataclass
class Detection3D:
    """单个 3D 检测结果（相机坐标系）"""
    detection_2d: Detection2D
    position_cam: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [Xc, Yc, Zc] m
    orientation_cam: np.ndarray = field(default_factory=lambda: np.eye(3)) # 3×3 旋转矩阵


@dataclass
class PoseTarget:
    """给手臂类使用的目标位姿（机械臂基坐标系）"""
    object_id: str
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))   # [X, Y, Z] m
    orientation: np.ndarray = field(default_factory=lambda: np.eye(3))   # 3×3 旋转矩阵
    confidence: float = 0.0
    timestamp_ns: int = 0
    is_valid: bool = False
    message: str = ""

    @property
    def quaternion(self) -> np.ndarray:
        """旋转矩阵 → 四元数 [x, y, z, w]"""
        R = self.orientation
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        return np.array([x, y, z, w], dtype=np.float64)


@dataclass
class HealthStatus:
    """系统健康状态"""
    cameras_ok: Dict[str, bool] = field(default_factory=dict)
    model_loaded: bool = False
    calibration_loaded: bool = False
    detection_fps: float = 0.0
    drop_frame_count: int = 0
    last_check_time: float = 0.0
    error_messages: List[str] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        return (
            all(self.cameras_ok.values())
            and self.model_loaded
            and self.calibration_loaded
        )


# ══════════════════════════════════════════════════════════════
# 检测器后端抽象（策略模式，方便替换模型框架）
# ══════════════════════════════════════════════════════════════

class BaseDetector:
    """检测器基类接口"""
    def detect(self, image: np.ndarray, roi: Optional[Tuple] = None) -> List[Detection2D]:
        raise NotImplementedError

    def load(self, model_path: str, **kwargs) -> bool:
        raise NotImplementedError

    @property
    def is_ready(self) -> bool:
        return False


class YOLODetector(BaseDetector):
    """YOLOv8 检测器（ultralytics 后端）"""

    def __init__(self):
        self._model = None
        self._model_path: str = ""
        self._conf_threshold: float = 0.5
        self._target_classes: Optional[List[str]] = None
        self._lock = threading.Lock()

    def load(self, model_path: str, conf: float = 0.5,
             target_classes: Optional[List[str]] = None, **kwargs) -> bool:
        try:
            from ultralytics import YOLO
            with self._lock:
                self._model = YOLO(model_path)
                self._model_path = model_path
                self._conf_threshold = conf
                self._target_classes = target_classes
            logger.info(f"YOLO 模型加载成功: {model_path}")
            return True
        except Exception as e:
            logger.error(f"YOLO 模型加载失败: {e}")
            return False

    def detect(self, image: np.ndarray, roi: Optional[Tuple] = None) -> List[Detection2D]:
        if not self.is_ready:
            return []

        now_ns = int(time.time() * 1e9)
        crop_offset = (0, 0)

        # ROI 裁剪
        if roi is not None and HAS_OPENCV:
            rx, ry, rw, rh = roi
            image = image[ry:ry + rh, rx:rx + rw]
            crop_offset = (rx, ry)

        results = []
        try:
            with self._lock:
                preds = self._model(image, conf=self._conf_threshold, verbose=False)

            for pred in preds:
                for box in pred.boxes:
                    cls_name = pred.names[int(box.cls)]
                    if self._target_classes and cls_name not in self._target_classes:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    # 还原 ROI 偏移
                    x1 += crop_offset[0]; x2 += crop_offset[0]
                    y1 += crop_offset[1]; y2 += crop_offset[1]
                    results.append(Detection2D(
                        object_id=cls_name,
                        confidence=float(box.conf),
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        timestamp_ns=now_ns,
                    ))
        except Exception as e:
            logger.warning(f"YOLO 推理出错: {e}")
        return results

    @property
    def is_ready(self) -> bool:
        return self._model is not None


class ONNXDetector(BaseDetector):
    """ONNX Runtime 检测器（更轻量，兼容 TensorRT EP）"""

    def __init__(self):
        self._session = None
        self._input_name: str = ""
        self._conf_threshold: float = 0.5
        self._nms_threshold: float = 0.45
        self._input_size: Tuple[int, int] = (640, 640)
        self._class_names: List[str] = []
        self._lock = threading.Lock()

    def load(self, model_path: str, conf: float = 0.5, nms: float = 0.45,
             class_names: Optional[List[str]] = None,
             use_tensorrt: bool = False, **kwargs) -> bool:
        try:
            import onnxruntime as ort
            providers = (["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
                         if use_tensorrt
                         else ["CUDAExecutionProvider", "CPUExecutionProvider"])
            with self._lock:
                self._session = ort.InferenceSession(model_path, providers=providers)
                self._input_name = self._session.get_inputs()[0].name
                _, _, h, w = self._session.get_inputs()[0].shape
                self._input_size = (int(h), int(w))
                self._conf_threshold = conf
                self._nms_threshold = nms
                self._class_names = class_names or []
            logger.info(f"ONNX 模型加载成功: {model_path}, input={self._input_size}")
            return True
        except Exception as e:
            logger.error(f"ONNX 模型加载失败: {e}")
            return False

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """标准 YOLOv8 ONNX 前处理：letterbox + normalize"""
        if not HAS_OPENCV:
            raise RuntimeError("ONNX 推理需要 OpenCV")
        h_in, w_in = self._input_size
        img = cv2.resize(image, (w_in, h_in))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis]  # NCHW
        return img

    def detect(self, image: np.ndarray, roi: Optional[Tuple] = None) -> List[Detection2D]:
        if not self.is_ready or not HAS_OPENCV:
            return []

        now_ns = int(time.time() * 1e9)
        orig_h, orig_w = image.shape[:2]
        crop_offset = (0, 0)

        if roi is not None:
            rx, ry, rw, rh = roi
            image = image[ry:ry + rh, rx:rx + rw]
            crop_offset = (rx, ry)
            orig_h, orig_w = rh, rw

        try:
            blob = self._preprocess(image)
            with self._lock:
                outputs = self._session.run(None, {self._input_name: blob})

            # outputs[0]: shape [1, num_classes+4, num_anchors]
            preds = outputs[0][0].T  # [num_anchors, num_classes+4]
            in_h, in_w = self._input_size
            scale_x, scale_y = orig_w / in_w, orig_h / in_h

            results: List[Detection2D] = []
            boxes, scores, class_ids = [], [], []

            for pred in preds:
                cx, cy, w, h = pred[:4]
                class_scores = pred[4:]
                cls_id = int(np.argmax(class_scores))
                conf = float(class_scores[cls_id])
                if conf < self._conf_threshold:
                    continue
                x1 = int((cx - w / 2) * scale_x) + crop_offset[0]
                y1 = int((cy - h / 2) * scale_y) + crop_offset[1]
                bw = int(w * scale_x)
                bh = int(h * scale_y)
                boxes.append([x1, y1, bw, bh])
                scores.append(conf)
                class_ids.append(cls_id)

            # NMS
            if boxes:
                indices = cv2.dnn.NMSBoxes(
                    boxes, scores, self._conf_threshold, self._nms_threshold
                )
                for i in (indices.flatten() if len(indices) else []):
                    x1, y1, bw, bh = boxes[i]
                    cls_name = (self._class_names[class_ids[i]]
                                if class_ids[i] < len(self._class_names)
                                else str(class_ids[i]))
                    results.append(Detection2D(
                        object_id=cls_name,
                        confidence=scores[i],
                        bbox=(x1, y1, x1 + bw, y1 + bh),
                        timestamp_ns=now_ns,
                    ))
            return results
        except Exception as e:
            logger.warning(f"ONNX 推理出错: {e}")
            return []

    @property
    def is_ready(self) -> bool:
        return self._session is not None


# ══════════════════════════════════════════════════════════════
# 主类：G2VisualPerception
# ══════════════════════════════════════════════════════════════

class G2VisualPerception:
    """
    智元 G2 机器人视觉感知中枢

    生命周期:
        vp = G2VisualPerception(config_path="vision_config.yaml")
        vp.init_cameras()
        vp.load_calibration()
        vp.load_model()
        # ─── 主循环 ───
        target = vp.get_execution_target("bottle")
        # 传给手臂类执行
        # ─────────────
        vp.safe_stop()
    """

    # ──────────────────────────────────────────
    # 构造 / 析构
    # ──────────────────────────────────────────

    def __init__(self, config_path: str = "vision_config.yaml",
                 stop_callback: Optional[Callable] = None):
        """
        Parameters
        ----------
        config_path : str
            YAML 配置文件路径，包含相机参数、标定文件路径、模型路径等。
        stop_callback : Callable, optional
            安全停止时回调，通常由手臂类注册，收到通知后暂停运动。
        """
        self._config_path = config_path
        self._stop_callback = stop_callback
        self._config: Dict[str, Any] = {}

        # ── 硬件句柄 ──
        self._camera: Optional[Any] = None          # agibot_gdk.Camera
        self._camera_lock = threading.Lock()

        # 各相机在线状态
        self._camera_online: Dict[str, bool] = {}
        # 重连线程
        self._reconnect_threads: Dict[str, threading.Thread] = {}

        # ── 标定数据 ──
        self._calib = CalibrationData()
        self._calib_lock = threading.RLock()

        # ── 检测模型 ──
        self._detector: Optional[BaseDetector] = None
        self._detector_lock = threading.Lock()
        self._model_path: str = ""
        self._model_mtime: float = 0.0  # 用于热加载检测

        # ── 图像缓冲区（线程安全） ──
        # key: CameraID.value, value: (image_ndarray, timestamp_ns)
        self._frame_buffer: Dict[str, Tuple[np.ndarray, int]] = {}
        self._frame_lock = threading.RLock()

        # ── 检测结果队列 ──
        self._detection_queue: queue.Queue = queue.Queue(maxsize=10)
        # 最新结果缓存
        self._latest_detections: List[Detection2D] = []
        self._detections_lock = threading.Lock()

        # ── 手眼标定数据采集 ──
        self._calib_dataset: List[Dict[str, Any]] = []
        self._calib_save_dir: str = "hand_eye_calib_data"

        # ── 训练样本 ──
        self._training_save_dir: str = "training_samples"

        # ── 监控统计 ──
        self._health = HealthStatus()
        self._stats_lock = threading.Lock()
        self._detect_timestamps: List[float] = []   # 最近检测时间戳（用于 FPS 计算）
        self._detect_success: int = 0
        self._detect_total: int = 0

        # ── 控制标志 ──
        self._running = False
        self._stop_event = threading.Event()

        # ── 热加载检查线程 ──
        self._hot_reload_thread: Optional[threading.Thread] = None

        # 加载配置
        self._load_config()
        logger.info("G2VisualPerception 实例创建完毕")

    def __del__(self):
        self.safe_stop()

    # ══════════════════════════════════════════
    # 模块一：系统初始化与配置
    # ══════════════════════════════════════════

    def _load_config(self) -> bool:
        """从 YAML 文件加载配置（内部方法）"""
        if not HAS_YAML:
            logger.warning("PyYAML 未安装，使用默认配置")
            self._config = self._default_config()
            return False

        path = Path(self._config_path)
        if not path.exists():
            logger.warning(f"配置文件不存在: {path}，使用默认配置并生成模板")
            self._config = self._default_config()
            self._save_default_config()
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
            logger.info(f"配置文件加载成功: {path}")
            return True
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            self._config = self._default_config()
            return False

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        return {
            "cameras": {
                "enabled": [
                    "kHeadColor",
                    "kHeadDepth",
                    "kHandLeftColor",
                    "kHandRightColor",
                ],
                "timeout_ms": 1000.0,
                "reconnect_interval_s": 3.0,
            },
            "calibration": {
                "file": "calibration.yaml",
            },
            "model": {
                "type": "yolo",          # "yolo" | "onnx"
                "path": "model.pt",
                "conf": 0.5,
                "nms": 0.45,
                "target_classes": ["bottle"],
                "hot_reload_interval_s": 5.0,
                "use_tensorrt": False,
            },
            "depth": {
                "method": "depth_camera",  # "depth_camera" | "stereo"
                "max_depth_m": 3.0,
                "min_depth_m": 0.1,
            },
            "grasp_offset": {
                "z_approach_m": 0.10,      # 物体上方接近偏移
            },
            "logging": {
                "log_file": "vision.log",
                "level": "INFO",
            },
        }

    def _save_default_config(self):
        """将默认配置保存为模板文件"""
        if not HAS_YAML:
            return
        try:
            with open(self._config_path, "w", encoding="utf-8") as f:
                yaml.dump(self._default_config(), f,
                          allow_unicode=True, default_flow_style=False)
            logger.info(f"已生成配置模板: {self._config_path}")
        except Exception as e:
            logger.warning(f"生成配置模板失败: {e}")

    def init_cameras(self) -> bool:
        """
        模块一 · 初始化相机
        枚举并初始化 G2 的所有已配置相机，设置初始帧缓冲，
        并启动断线重连守护线程。

        Returns
        -------
        bool : 至少一个相机成功初始化则返回 True
        """
        if not HAS_GDK:
            logger.warning("GDK 未加载，跳过相机初始化（Mock 模式）")
            for cam_id in self._config["cameras"]["enabled"]:
                self._camera_online[cam_id] = False
            return False

        logger.info("正在初始化相机...")
        try:
            with self._camera_lock:
                self._camera = agibot_gdk.Camera()
            time.sleep(2.0)  # 等待 SDK 内部枚举完成
        except Exception as e:
            logger.error(f"相机句柄创建失败: {e}")
            return False

        enabled = self._config["cameras"]["enabled"]
        success_count = 0
        for cam_id_str in enabled:
            ok = self._probe_camera(cam_id_str)
            self._camera_online[cam_id_str] = ok
            self._health.cameras_ok[cam_id_str] = ok
            if ok:
                success_count += 1
                logger.info(f"  ✅ 相机在线: {cam_id_str}")
            else:
                logger.warning(f"  ❌ 相机离线: {cam_id_str}，启动重连线程")
                self._start_reconnect_thread(cam_id_str)

        self._running = True
        self._stop_event.clear()
        logger.info(f"相机初始化完成 ({success_count}/{len(enabled)} 在线)")
        return success_count > 0

    def _probe_camera(self, cam_id_str: str, timeout_ms: float = 500.0) -> bool:
        """尝试从指定相机获取一帧，验证是否在线"""
        try:
            cam_type = getattr(agibot_gdk.CameraType, cam_id_str, None)
            if cam_type is None:
                return False
            with self._camera_lock:
                img = self._camera.get_latest_image(cam_type, timeout_ms)
            if img is not None:
                self._update_frame_buffer(cam_id_str, img)
                return True
        except Exception:
            pass
        return False

    def _start_reconnect_thread(self, cam_id_str: str):
        """为指定相机启动断线重连守护线程"""
        if cam_id_str in self._reconnect_threads:
            t = self._reconnect_threads[cam_id_str]
            if t.is_alive():
                return  # 已有重连线程在运行

        interval = self._config["cameras"].get("reconnect_interval_s", 3.0)

        def _reconnect_loop():
            while not self._stop_event.is_set():
                time.sleep(interval)
                if self._camera_online.get(cam_id_str, False):
                    break  # 已恢复，退出
                ok = self._probe_camera(cam_id_str)
                if ok:
                    self._camera_online[cam_id_str] = True
                    self._health.cameras_ok[cam_id_str] = True
                    logger.info(f"  🔄 相机重连成功: {cam_id_str}")
                    break

        t = threading.Thread(target=_reconnect_loop, daemon=True,
                             name=f"reconnect_{cam_id_str}")
        t.start()
        self._reconnect_threads[cam_id_str] = t

    def _update_frame_buffer(self, cam_id_str: str, gdk_image):
        """将 GDK 图像解码后存入线程安全帧缓冲"""
        if not HAS_OPENCV:
            return
        try:
            arr = self._decode_gdk_image(gdk_image)
            if arr is not None:
                ts = getattr(gdk_image, "timestamp_ns", int(time.time() * 1e9))
                with self._frame_lock:
                    self._frame_buffer[cam_id_str] = (arr, ts)
        except Exception as e:
            logger.debug(f"帧缓冲更新失败 ({cam_id_str}): {e}")

    @staticmethod
    def _decode_gdk_image(image) -> Optional[np.ndarray]:
        """将 agibot_gdk 图像对象解码为 numpy BGR/depth 数组"""
        if not HAS_OPENCV or image is None:
            return None
        try:
            enc = image.encoding
            if enc == agibot_gdk.Encoding.JPEG or enc == agibot_gdk.Encoding.PNG:
                nparr = np.frombuffer(image.data, np.uint8)
                return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            elif enc == agibot_gdk.Encoding.UNCOMPRESSED:
                fmt = image.color_format
                if fmt == agibot_gdk.ColorFormat.RGB:
                    arr = np.frombuffer(image.data, np.uint8).reshape(
                        image.height, image.width, 3)
                    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                elif fmt == agibot_gdk.ColorFormat.BGR:
                    return np.frombuffer(image.data, np.uint8).reshape(
                        image.height, image.width, 3)
                elif fmt == agibot_gdk.ColorFormat.GRAY8:
                    arr = np.frombuffer(image.data, np.uint8).reshape(
                        image.height, image.width)
                    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                elif fmt in (agibot_gdk.ColorFormat.GRAY16,
                             agibot_gdk.ColorFormat.RS2_FORMAT_Z16):
                    # 深度图：uint16，单位 mm
                    return np.frombuffer(image.data, np.uint16).reshape(
                        image.height, image.width).copy()
        except Exception:
            pass
        return None

    # ──────────────────────────────────────────

    def load_calibration(self, calib_path: Optional[str] = None) -> bool:
        """
        模块一 · 加载标定数据

        从 .yaml 或 .json 文件加载相机内参、畸变系数、
        手眼变换矩阵（T_cam2base 或 T_cam2tool）。

        Parameters
        ----------
        calib_path : str, optional
            标定文件路径；为 None 时从配置读取。

        Returns
        -------
        bool : 加载成功返回 True
        """
        path = calib_path or self._config.get("calibration", {}).get(
            "file", "calibration.yaml")
        path = Path(path)

        if not path.exists():
            logger.warning(f"标定文件不存在: {path}，请先执行手眼标定")
            return False

        try:
            if path.suffix in (".yaml", ".yml"):
                if not HAS_YAML:
                    raise RuntimeError("需要 PyYAML")
                with open(path, "r") as f:
                    data: Dict = yaml.safe_load(f)
            else:
                with open(path, "r") as f:
                    data = json.load(f)

            with self._calib_lock:
                calib = CalibrationData()

                # 内参
                for cam_name, intr in data.get("intrinsics", {}).items():
                    calib.intrinsics[cam_name] = CameraIntrinsics(
                        fx=intr["fx"], fy=intr["fy"],
                        cx=intr["cx"], cy=intr["cy"],
                        width=intr.get("width", 0),
                        height=intr.get("height", 0),
                        dist_coeffs=np.array(intr.get("dist_coeffs", [0]*5)),
                    )

                # 手眼矩阵
                if "T_cam2base" in data:
                    calib.T_cam2base = np.array(data["T_cam2base"], dtype=np.float64)
                if "T_cam2tool" in data:
                    calib.T_cam2tool = np.array(data["T_cam2tool"], dtype=np.float64)

                calib.stereo_baseline = data.get("stereo_baseline", 0.065)
                calib.source_path = str(path)
                calib.load_time = time.time()
                self._calib = calib

            self._health.calibration_loaded = True
            logger.info(f"标定数据加载成功: {path}")
            logger.info(f"  内参相机数: {len(self._calib.intrinsics)}")
            logger.info(f"  T_cam2base: {'已加载' if self._calib.T_cam2base is not None else '无'}")
            return True

        except Exception as e:
            logger.error(f"标定数据加载失败: {e}")
            return False

    # ──────────────────────────────────────────

    def load_model(self, model_path: Optional[str] = None,
                   model_type: Optional[str] = None, **kwargs) -> bool:
        """
        模块一 · 加载检测模型（支持热加载）

        Parameters
        ----------
        model_path : str, optional
            模型文件路径；为 None 时从配置读取。
        model_type : str, optional
            "yolo" 或 "onnx"；为 None 时从配置读取。
        **kwargs
            传递给具体检测器的额外参数（conf、nms、target_classes 等）。

        Returns
        -------
        bool : 加载成功返回 True
        """
        cfg_model = self._config.get("model", {})
        path = model_path or cfg_model.get("path", "model.pt")
        mtype = model_type or cfg_model.get("type", "yolo")

        # 合并配置参数（kwargs 优先）
        params = {
            "conf":           cfg_model.get("conf", 0.5),
            "nms":            cfg_model.get("nms", 0.45),
            "target_classes": cfg_model.get("target_classes"),
            "use_tensorrt":   cfg_model.get("use_tensorrt", False),
        }
        params.update(kwargs)

        if mtype == "yolo":
            detector = YOLODetector()
        elif mtype == "onnx":
            detector = ONNXDetector()
        else:
            logger.error(f"未知模型类型: {mtype}")
            return False

        ok = detector.load(path, **params)
        if ok:
            with self._detector_lock:
                self._detector = detector
                self._model_path = path
                self._model_mtime = os.path.getmtime(path) if os.path.exists(path) else 0.0
            self._health.model_loaded = True

            # 启动热加载检查线程
            reload_interval = cfg_model.get("hot_reload_interval_s", 5.0)
            if reload_interval > 0:
                self._start_hot_reload_thread(reload_interval)

        return ok

    def _start_hot_reload_thread(self, interval: float):
        """启动模型热加载检测线程"""
        if (self._hot_reload_thread and self._hot_reload_thread.is_alive()):
            return

        def _watch():
            while not self._stop_event.is_set():
                time.sleep(interval)
                path = self._model_path
                if not path or not os.path.exists(path):
                    continue
                mtime = os.path.getmtime(path)
                if mtime > self._model_mtime:
                    logger.info(f"检测到模型文件更新，热加载: {path}")
                    self.load_model(path)

        self._hot_reload_thread = threading.Thread(
            target=_watch, daemon=True, name="model_hot_reload")
        self._hot_reload_thread.start()

    # ══════════════════════════════════════════
    # 模块二：数据采集与手眼标定
    # ══════════════════════════════════════════

    def capture_sync_frame(self, cam_id: str = "kHeadColor",
                           robot=None) -> Optional[Dict[str, Any]]:
        """
        模块二 · 同步采集图像与机械臂位姿

        Parameters
        ----------
        cam_id : str
            相机 ID（CameraType 名称字符串）
        robot : agibot_gdk.Robot, optional
            机械臂对象，用于同步读取末端位姿。
            若为 None 则不读取机械臂状态。

        Returns
        -------
        dict with keys:
            "image"      : np.ndarray (BGR)
            "timestamp_ns": int
            "robot_pose" : dict | None  (末端位姿)
        """
        if not HAS_GDK or self._camera is None:
            logger.warning("相机未初始化，无法采集同步帧")
            return None

        try:
            cam_type = getattr(agibot_gdk.CameraType, cam_id)
            timeout = self._config["cameras"]["timeout_ms"]
            with self._camera_lock:
                gdk_img = self._camera.get_latest_image(cam_type, timeout)

            if gdk_img is None:
                logger.warning(f"采集帧失败: {cam_id}")
                return None

            img_arr = self._decode_gdk_image(gdk_img)
            ts = gdk_img.timestamp_ns

            robot_pose = None
            if robot is not None:
                try:
                    joint_states = robot.get_joint_states()
                    # 也可扩展为读取 end_effector_pose
                    robot_pose = {
                        "joint_states": joint_states,
                        "timestamp":    ts,
                    }
                except Exception as e:
                    logger.warning(f"读取机械臂状态失败: {e}")

            return {"image": img_arr, "timestamp_ns": ts, "robot_pose": robot_pose}

        except Exception as e:
            logger.error(f"capture_sync_frame 出错: {e}")
            return None

    def save_calibration_data(self, image: np.ndarray, robot_pose: Dict[str, Any],
                              sample_id: Optional[int] = None) -> str:
        """
        模块二 · 保存手眼标定样本

        将同步采集的图像和机械臂位姿保存到数据集目录。

        Returns
        -------
        str : 保存目录路径
        """
        os.makedirs(self._calib_save_dir, exist_ok=True)
        idx = sample_id if sample_id is not None else len(self._calib_dataset)
        img_path = os.path.join(self._calib_save_dir, f"frame_{idx:04d}.png")
        pose_path = os.path.join(self._calib_save_dir, f"pose_{idx:04d}.json")

        if HAS_OPENCV and image is not None:
            cv2.imwrite(img_path, image)

        with open(pose_path, "w") as f:
            # numpy 不可直接序列化，转 list
            def _serialize(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            serializable_pose = {k: _serialize(v) for k, v in robot_pose.items()}
            json.dump(serializable_pose, f, indent=2)

        record = {"index": idx, "image": img_path, "pose": pose_path}
        self._calib_dataset.append(record)
        logger.info(f"标定样本已保存 #{idx}: {img_path}")
        return self._calib_save_dir

    def run_hand_eye_calibration(self,
                                 calib_type: str = "eye_to_hand",
                                 output_path: str = "calibration.yaml") -> Optional[np.ndarray]:
        """
        模块二 · 执行手眼标定

        Parameters
        ----------
        calib_type : str
            "eye_to_hand"  → 固定基底相机，输出 T_cam2base
            "eye_in_hand"  → 末端相机，输出 T_cam2tool
        output_path : str
            计算结果写入的标定文件路径

        Returns
        -------
        np.ndarray : 4×4 手眼变换矩阵，失败返回 None

        Notes
        -----
        需要预先采集 ≥15 对样本（标定板图像 + 机械臂位姿），
        通过 capture_sync_frame() + save_calibration_data() 完成。
        """
        if not HAS_OPENCV:
            logger.error("手眼标定需要 OpenCV")
            return None

        logger.info(f"开始手眼标定，样本数: {len(self._calib_dataset)}，类型: {calib_type}")
        if len(self._calib_dataset) < 10:
            logger.warning("样本数量不足（建议 ≥15），标定精度可能较低")

        # ── 检测标定板角点 ──
        board_size = (9, 6)   # 内角点行列数，可从配置读取
        square_size = 0.025   # 棋盘格边长（m）
        objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        objp *= square_size

        R_gripper2base_list, t_gripper2base_list = [], []
        R_target2cam_list,   t_target2cam_list   = [], []

        for record in self._calib_dataset:
            img = cv2.imread(record["image"])
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, board_size, None)
            if not found:
                logger.debug(f"跳过样本 #{record['index']}：未检测到棋盘角点")
                continue

            # 亚像素精化
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            # 读取当前相机内参
            cam_name = self._config["cameras"]["enabled"][0]
            with self._calib_lock:
                intr = self._calib.intrinsics.get(cam_name)
            if intr is None:
                logger.error("未找到相机内参，无法进行标定")
                return None

            # PnP 求解棋盘→相机变换
            ret, rvec, tvec = cv2.solvePnP(
                objp, corners, intr.K, intr.dist_coeffs)
            if not ret:
                continue
            R_board, _ = cv2.Rodrigues(rvec)
            R_target2cam_list.append(R_board)
            t_target2cam_list.append(tvec.flatten())

            # 读取机械臂位姿
            with open(record["pose"], "r") as f:
                pose_data = json.load(f)
            # 此处期望 pose_data 包含 "end_effector" 的旋转矩阵和平移向量
            # 实际场景中需根据 robot.get_end_effector_pose() 的返回格式适配
            # 示例：假设已解析为 4×4 矩阵
            T_tool2base = np.array(pose_data.get("T_tool2base",
                                                  np.eye(4).tolist()))
            R_gripper2base_list.append(T_tool2base[:3, :3])
            t_gripper2base_list.append(T_tool2base[:3, 3])

        if len(R_target2cam_list) < 3:
            logger.error("有效标定样本不足（至少需要3对）")
            return None

        # ── 调用 OpenCV 手眼标定 ──
        method = cv2.CALIB_HAND_EYE_TSAI
        R_cam2tool, t_cam2tool = cv2.calibrateHandEye(
            R_gripper2base_list, t_gripper2base_list,
            R_target2cam_list,   t_target2cam_list,
            method=method,
        )

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_cam2tool
        T[:3, 3]  = t_cam2tool.flatten()

        # ── 写入标定文件 ──
        with self._calib_lock:
            if calib_type == "eye_to_hand":
                self._calib.T_cam2base = T
            else:
                self._calib.T_cam2tool = T
            self._health.calibration_loaded = True

        # 序列化保存
        calib_dict: Dict[str, Any] = {}
        if HAS_YAML:
            # 读取已有文件合并
            p = Path(output_path)
            if p.exists():
                with open(p) as f:
                    calib_dict = yaml.safe_load(f) or {}
            key = "T_cam2base" if calib_type == "eye_to_hand" else "T_cam2tool"
            calib_dict[key] = T.tolist()
            with open(output_path, "w") as f:
                yaml.dump(calib_dict, f, allow_unicode=True)

        logger.info(f"手眼标定完成，结果已写入: {output_path}")
        logger.info(f"T_cam2{'base' if calib_type == 'eye_to_hand' else 'tool'}:\n{T}")
        return T

    # ══════════════════════════════════════════
    # 模块三：视觉检测与处理
    # ══════════════════════════════════════════

    def get_latest_frame(self, cam_id: str = "kHeadColor") -> Optional[Tuple[np.ndarray, int]]:
        """
        从帧缓冲获取最新图像帧（不触发新采集）。

        Returns
        -------
        (image_ndarray, timestamp_ns) | None
        """
        # 先尝试从 GDK 实时拉取更新帧缓冲
        if HAS_GDK and self._camera is not None:
            try:
                cam_type = getattr(agibot_gdk.CameraType, cam_id)
                timeout = self._config["cameras"]["timeout_ms"]
                with self._camera_lock:
                    gdk_img = self._camera.get_latest_image(cam_type, timeout)
                if gdk_img is not None:
                    self._update_frame_buffer(cam_id, gdk_img)
                    if not self._camera_online.get(cam_id, False):
                        self._camera_online[cam_id] = True
                        self._health.cameras_ok[cam_id] = True
                        logger.info(f"相机恢复在线: {cam_id}")
            except Exception as e:
                # 相机掉线：标记并启动重连
                if self._camera_online.get(cam_id, True):
                    self._camera_online[cam_id] = False
                    self._health.cameras_ok[cam_id] = False
                    logger.warning(f"相机离线: {cam_id} — {e}")
                    self._start_reconnect_thread(cam_id)

        with self._frame_lock:
            return self._frame_buffer.get(cam_id)

    def detect_objects(self,
                       cam_id: str = "kHeadColor",
                       roi: Optional[Tuple[int, int, int, int]] = None,
                       ) -> List[Detection2D]:
        """
        模块三 · 目标检测

        在指定相机的最新帧上执行推理，返回检测结果列表。

        Parameters
        ----------
        cam_id : str
            使用的相机 ID
        roi : (x, y, w, h) | None
            仅在 ROI 区域内推理（像素坐标），None 表示全图

        Returns
        -------
        List[Detection2D]
        """
        frame_data = self.get_latest_frame(cam_id)
        if frame_data is None:
            logger.debug(f"detect_objects: 无法获取帧 ({cam_id})")
            return []

        image, ts = frame_data

        with self._detector_lock:
            detector = self._detector

        if detector is None or not detector.is_ready:
            logger.warning("检测器未加载")
            return []

        self._detect_total += 1
        t0 = time.time()

        try:
            detections = detector.detect(image, roi=roi)
            # 补充时间戳
            for d in detections:
                if d.timestamp_ns == 0:
                    d.timestamp_ns = ts

            elapsed = time.time() - t0
            self._detect_success += 1

            # 更新 FPS 统计
            with self._stats_lock:
                self._detect_timestamps.append(time.time())
                # 保留最近 2 秒的时间戳
                cutoff = time.time() - 2.0
                self._detect_timestamps = [t for t in self._detect_timestamps if t > cutoff]
                self._health.detection_fps = len(self._detect_timestamps) / 2.0

            # 更新最新结果缓存
            with self._detections_lock:
                self._latest_detections = copy.deepcopy(detections)

            logger.debug(f"检测完成: {len(detections)} 个目标, 耗时 {elapsed*1000:.1f}ms")
            return detections

        except Exception as e:
            logger.warning(f"检测推理出错: {e}")
            return []

    # ══════════════════════════════════════════
    # 模块三 · 3D 位姿估计
    # ══════════════════════════════════════════

    def estimate_3d_pose(self, det2d: Detection2D,
                         color_cam_id: str = "kHeadColor",
                         depth_cam_id: str = "kHeadDepth",
                         ) -> Optional[Detection3D]:
        """
        模块三 · 2D → 3D 坐标估计

        根据检测框中心像素坐标，结合深度图（或双目）
        估计物体在相机坐标系下的 3D 位置。

        Parameters
        ----------
        det2d : Detection2D
            2D 检测结果
        color_cam_id : str
            彩色相机 ID（内参来源）
        depth_cam_id : str
            深度相机 ID

        Returns
        -------
        Detection3D | None
        """
        method = self._config.get("depth", {}).get("method", "depth_camera")

        if method == "depth_camera":
            return self._estimate_3d_from_depth(det2d, color_cam_id, depth_cam_id)
        elif method == "stereo":
            return self._estimate_3d_from_stereo(det2d)
        else:
            logger.error(f"未知的深度估计方法: {method}")
            return None

    def _estimate_3d_from_depth(self, det2d: Detection2D,
                                 color_cam_id: str,
                                 depth_cam_id: str) -> Optional[Detection3D]:
        """利用深度图估计 3D 坐标"""
        # 获取深度帧
        depth_data = self.get_latest_frame(depth_cam_id)
        if depth_data is None:
            logger.debug("深度帧不可用")
            return None

        depth_map, _ = depth_data
        if depth_map.ndim != 2:
            # 如果深度图被转成 BGR，取单通道
            if depth_map.ndim == 3:
                depth_map = depth_map[:, :, 0].astype(np.uint16)
            else:
                return None

        # 边界框中心像素
        x1, y1, x2, y2 = det2d.bbox
        cx_px = (x1 + x2) // 2
        cy_px = (y1 + y2) // 2

        # 深度采样：取 bbox 中心区域均值（更鲁棒）
        region_half = max(3, (x2 - x1) // 8)
        rx1 = max(0, cx_px - region_half)
        rx2 = min(depth_map.shape[1], cx_px + region_half)
        ry1 = max(0, cy_px - region_half)
        ry2 = min(depth_map.shape[0], cy_px + region_half)

        depth_region = depth_map[ry1:ry2, rx1:rx2].astype(np.float32)
        valid_mask = (depth_region > 100) & (depth_region < 5000)  # 单位 mm
        if not np.any(valid_mask):
            logger.debug(f"目标 '{det2d.object_id}' 深度无效")
            return None

        Z_mm = float(np.median(depth_region[valid_mask]))
        Z_m  = Z_mm / 1000.0

        # 深度范围过滤
        cfg_depth = self._config.get("depth", {})
        if not (cfg_depth.get("min_depth_m", 0.05) <= Z_m <= cfg_depth.get("max_depth_m", 3.5)):
            logger.debug(f"深度超出范围: {Z_m:.3f}m")
            return None

        # 相机内参反投影
        with self._calib_lock:
            intr = self._calib.intrinsics.get(color_cam_id)
        if intr is None:
            logger.warning(f"未找到相机内参: {color_cam_id}")
            return None

        X_m = (cx_px - intr.cx) * Z_m / intr.fx
        Y_m = (cy_px - intr.cy) * Z_m / intr.fy

        position = np.array([X_m, Y_m, Z_m], dtype=np.float64)
        return Detection3D(detection_2d=det2d, position_cam=position)

    def _estimate_3d_from_stereo(self, det2d: Detection2D) -> Optional[Detection3D]:
        """利用双目视差估计 3D 坐标（简化实现，需 stereo_baseline 和双目内参）"""
        left_data  = self.get_latest_frame("kHeadStereoLeft")
        right_data = self.get_latest_frame("kHeadStereoRight")
        if left_data is None or right_data is None:
            return None

        if not HAS_OPENCV:
            return None

        left_img,  _ = left_data
        right_img, _ = right_data

        # 计算稠密视差图（SGBM 算法）
        gray_l = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        stereo = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=128, blockSize=5,
            P1=8 * 3 * 5 ** 2, P2=32 * 3 * 5 ** 2,
            disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=32,
        )
        disparity = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0

        # bbox 中心视差
        x1, y1, x2, y2 = det2d.bbox
        cx_px = (x1 + x2) // 2
        cy_px = (y1 + y2) // 2
        disp_val = float(disparity[cy_px, cx_px]) if disparity[cy_px, cx_px] > 0 else 0

        if disp_val <= 0:
            return None

        with self._calib_lock:
            intr = self._calib.intrinsics.get("kHeadStereoLeft")
            baseline = self._calib.stereo_baseline
        if intr is None:
            return None

        Z_m = intr.fx * baseline / disp_val
        X_m = (cx_px - intr.cx) * Z_m / intr.fx
        Y_m = (cy_px - intr.cy) * Z_m / intr.fy

        position = np.array([X_m, Y_m, Z_m], dtype=np.float64)
        return Detection3D(detection_2d=det2d, position_cam=position)

    # ══════════════════════════════════════════
    # 模块四：坐标变换与高级接口
    # ══════════════════════════════════════════

    def transform_cam_to_base(self, position_cam: np.ndarray,
                               orientation_cam: Optional[np.ndarray] = None,
                               ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        模块四 · 相机坐标系 → 机械臂基坐标系

        Parameters
        ----------
        position_cam : np.ndarray
            相机坐标系下的 3D 点 [X, Y, Z]（米）
        orientation_cam : np.ndarray | None
            相机坐标系下的 3×3 旋转矩阵（可选）

        Returns
        -------
        (position_base, orientation_base) | None
            机械臂基坐标系下的位置和姿态
        """
        with self._calib_lock:
            T = self._calib.T_cam2base

        if T is None:
            logger.error("T_cam2base 未加载，无法进行坐标变换")
            return None

        # 齐次坐标变换
        p_hom = np.array([*position_cam, 1.0], dtype=np.float64)
        p_base = (T @ p_hom)[:3]

        # 旋转变换
        R_cam2base = T[:3, :3]
        if orientation_cam is not None:
            R_base = R_cam2base @ orientation_cam
        else:
            R_base = R_cam2base.copy()

        return p_base, R_base

    def get_execution_target(self,
                              object_id: str,
                              cam_id: str = "kHeadColor",
                              depth_cam_id: str = "kHeadDepth",
                              roi: Optional[Tuple] = None,
                              grasp_offset: Optional[np.ndarray] = None,
                              ) -> PoseTarget:
        """
        模块四 · 高级接口：给手臂类的目标位姿

        一站式完成：检测 → 3D 估计 → 坐标变换 → 抓取偏移。
        手臂类只需调用此方法，获取结果后执行 MoveL 即可。

        Parameters
        ----------
        object_id : str
            目标物体类别（如 "bottle"）
        cam_id : str
            检测相机
        depth_cam_id : str
            深度相机
        roi : (x, y, w, h) | None
            检测 ROI
        grasp_offset : np.ndarray | None
            抓取额外偏移量 [dx, dy, dz]（基坐标系下，米）；
            为 None 时从配置读取 z_approach_m

        Returns
        -------
        PoseTarget
            is_valid=True 时结果可用；is_valid=False 时 message 说明原因
        """
        # ── 步骤1：检测 ──
        detections = self.detect_objects(cam_id=cam_id, roi=roi)
        candidates = [d for d in detections if d.object_id == object_id]

        if not candidates:
            return PoseTarget(object_id=object_id,
                              is_valid=False, message=f"未检测到目标: {object_id}")

        # 取置信度最高的
        best = max(candidates, key=lambda d: d.confidence)

        # ── 步骤2：3D 估计 ──
        det3d = self.estimate_3d_pose(best, color_cam_id=cam_id, depth_cam_id=depth_cam_id)
        if det3d is None:
            return PoseTarget(object_id=object_id,
                              is_valid=False, message="3D 坐标估计失败（深度数据无效）")

        # ── 步骤3：坐标变换 ──
        result = self.transform_cam_to_base(det3d.position_cam, det3d.orientation_cam)
        if result is None:
            return PoseTarget(object_id=object_id,
                              is_valid=False, message="坐标变换失败（手眼标定未加载）")

        position_base, orientation_base = result

        # ── 步骤4：抓取偏移 ──
        if grasp_offset is None:
            z_off = self._config.get("grasp_offset", {}).get("z_approach_m", 0.10)
            grasp_offset = np.array([0.0, 0.0, z_off])
        position_base = position_base + grasp_offset

        target = PoseTarget(
            object_id=object_id,
            position=position_base,
            orientation=orientation_base,
            confidence=best.confidence,
            timestamp_ns=best.timestamp_ns,
            is_valid=True,
            message="OK",
        )
        logger.info(f"get_execution_target → {object_id}: "
                    f"pos={position_base.round(4)}, conf={best.confidence:.2f}")
        return target

    # ══════════════════════════════════════════
    # 模块五：数据管理与模型训练
    # ══════════════════════════════════════════

    def collect_training_sample(self, label: str,
                                 cam_id: str = "kHeadColor",
                                 robot=None,
                                 annotation: Optional[Dict] = None) -> bool:
        """
        模块五 · 采集训练样本

        保存当前帧及对应标注信息（用于后续重新训练检测模型）。

        Parameters
        ----------
        label : str
            样本标签（物体类别名称）
        cam_id : str
            采集相机
        robot : agibot_gdk.Robot, optional
            用于记录采集时机械臂位姿
        annotation : dict, optional
            额外的标注数据（bbox、关键点等）

        Returns
        -------
        bool : 采集成功返回 True
        """
        frame_data = self.capture_sync_frame(cam_id=cam_id, robot=robot)
        if frame_data is None:
            logger.warning("collect_training_sample: 帧采集失败")
            return False

        os.makedirs(self._training_save_dir, exist_ok=True)
        ts = frame_data["timestamp_ns"]
        img_filename = f"{label}_{ts}.png"
        meta_filename = f"{label}_{ts}.json"

        img_path  = os.path.join(self._training_save_dir, img_filename)
        meta_path = os.path.join(self._training_save_dir, meta_filename)

        if HAS_OPENCV and frame_data["image"] is not None:
            cv2.imwrite(img_path, frame_data["image"])

        meta = {
            "label":      label,
            "image":      img_filename,
            "timestamp_ns": ts,
            "annotation": annotation or {},
        }
        if frame_data["robot_pose"]:
            meta["robot_pose"] = str(frame_data["robot_pose"])  # 简单序列化

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"训练样本已保存: {img_path} (label={label})")
        return True

    def start_training_job(self, config: Optional[Dict] = None) -> bool:
        """
        模块五 · 触发离线训练流程

        在独立子进程中执行训练脚本，不阻塞主控制线程。

        Parameters
        ----------
        config : dict, optional
            训练配置（会以 JSON 传递给训练脚本）

        Returns
        -------
        bool : 进程启动成功返回 True
        """
        train_script = (config or {}).get("script", "train.py")
        if not os.path.exists(train_script):
            logger.warning(f"训练脚本不存在: {train_script}")
            return False

        config_str = json.dumps(config or {})

        def _run():
            try:
                proc = subprocess.Popen(
                    ["python3", train_script, "--config", config_str],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                logger.info(f"训练进程已启动，PID={proc.pid}")
                stdout, stderr = proc.communicate()
                if proc.returncode == 0:
                    logger.info("训练完成")
                else:
                    logger.error(f"训练失败: {stderr.decode()[:200]}")
            except Exception as e:
                logger.error(f"启动训练进程失败: {e}")

        t = threading.Thread(target=_run, daemon=True, name="training_job")
        t.start()
        return True

    # ══════════════════════════════════════════
    # 模块六：监控与异常处理
    # ══════════════════════════════════════════

    def check_health(self) -> HealthStatus:
        """
        模块六 · 心跳健康检查

        检查各相机在线状态、模型就绪状态、标定加载状态，
        更新并返回 HealthStatus。

        Returns
        -------
        HealthStatus
        """
        # 探测相机
        enabled = self._config["cameras"].get("enabled", [])
        for cam_id in enabled:
            is_ok = self._camera_online.get(cam_id, False)
            self._health.cameras_ok[cam_id] = is_ok

        # 检测器状态
        with self._detector_lock:
            self._health.model_loaded = (
                self._detector is not None and self._detector.is_ready)

        # 标定状态
        with self._calib_lock:
            self._health.calibration_loaded = (
                self._calib.T_cam2base is not None
                or self._calib.T_cam2tool is not None
            )

        # FPS（由 detect_objects 更新）
        self._health.last_check_time = time.time()

        if not self._health.is_healthy:
            msgs = []
            for cam_id, ok in self._health.cameras_ok.items():
                if not ok:
                    msgs.append(f"相机离线: {cam_id}")
            if not self._health.model_loaded:
                msgs.append("检测模型未加载")
            if not self._health.calibration_loaded:
                msgs.append("手眼标定未加载")
            self._health.error_messages = msgs

        return self._health

    def log_status(self) -> Dict[str, Any]:
        """
        模块六 · 记录并返回运行状态日志

        Returns
        -------
        dict : 包含检测频率、成功率、相机状态等统计信息
        """
        health = self.check_health()
        success_rate = (self._detect_success / max(1, self._detect_total)) * 100

        status = {
            "is_healthy":       health.is_healthy,
            "cameras":          health.cameras_ok,
            "model_loaded":     health.model_loaded,
            "calib_loaded":     health.calibration_loaded,
            "detection_fps":    round(health.detection_fps, 2),
            "detect_total":     self._detect_total,
            "detect_success":   self._detect_success,
            "success_rate_pct": round(success_rate, 1),
            "error_messages":   health.error_messages,
            "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 输出到日志
        level = logging.WARNING if not health.is_healthy else logging.INFO
        logger.log(level, f"状态报告: FPS={status['detection_fps']}, "
                          f"成功率={status['success_rate_pct']}%, "
                          f"健康={status['is_healthy']}")
        if health.error_messages:
            for msg in health.error_messages:
                logger.warning(f"  ⚠️  {msg}")

        return status

    def safe_stop(self):
        """
        模块六 · 安全停止

        1. 停止所有后台线程。
        2. 关闭相机连接。
        3. 通知外部手臂类暂停（通过注册的回调）。
        """
        if not self._running:
            return

        logger.info("正在安全停止 G2VisualPerception...")
        self._running = False
        self._stop_event.set()

        # 通知手臂类
        if self._stop_callback is not None:
            try:
                self._stop_callback()
                logger.info("已通知手臂类停止")
            except Exception as e:
                logger.warning(f"通知手臂类失败: {e}")

        # 关闭相机
        if HAS_GDK and self._camera is not None:
            try:
                with self._camera_lock:
                    self._camera.close_camera()
                logger.info("相机已关闭")
            except Exception as e:
                logger.warning(f"关闭相机时出错: {e}")

        logger.info("G2VisualPerception 已停止")


# ══════════════════════════════════════════════════════════════
# 使用示例（配合手臂类）
# ══════════════════════════════════════════════════════════════

def demo_usage():
    """
    演示如何与手臂类配合使用。

    手臂类（ArmController）需要实现：
      - pause() → None
      - move_to_pose(position, orientation) → None
    """

    # ── 手臂类停止回调（解耦） ──
    def arm_pause_callback():
        print("[ArmController] 收到视觉停止信号，手臂暂停")

    # ── 初始化视觉感知模块 ──
    vp = G2VisualPerception(
        config_path="vision_config.yaml",
        stop_callback=arm_pause_callback,
    )

    # ── 生命周期：初始化 ──
    vp.init_cameras()
    vp.load_calibration("calibration.yaml")
    vp.load_model("yolov8n.pt", model_type="yolo",
                  target_classes=["bottle"], conf=0.5)

    # ── 主循环示例 ──
    try:
        while True:
            # 健康检查
            status = vp.log_status()
            if not status["is_healthy"]:
                print("系统异常，等待恢复...")
                time.sleep(1.0)
                continue

            # 获取目标（手臂类只需这一行）
            target = vp.get_execution_target(
                object_id="bottle",
                cam_id="kHeadColor",
                depth_cam_id="kHeadDepth",
            )

            if target.is_valid:
                print(f"目标位置（基坐标系）: {target.position}")
                print(f"目标四元数:           {target.quaternion}")
                # arm.move_to_pose(target.position, target.quaternion)
            else:
                print(f"目标不可用: {target.message}")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        vp.safe_stop()


if __name__ == "__main__":
    demo_usage()