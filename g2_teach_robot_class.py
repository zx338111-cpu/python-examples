#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G2 教学/搬运控制类封装

用途：
- 双臂关节点到点（PTP）
- 双臂末端直线点到点（LINE）
- 腰部读取与小步微调
- 夹爪控制
- 点位记录/加载/执行
- “收手 + 腰部小步回直”的运输姿态调整

设计说明：
- 双臂关节动作优先使用 move_arm_joint，并做分段与到位轮询
- 末端直线动作使用 end_effector_pose_control，按 50Hz 连续插值发送
- 腰部动作使用 move_waist_joint；建议只做小步试探，不建议直接大幅“回零”
- 点位以 JSON 原子写入保存
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import agibot_gdk  # type: ignore
    HAS_GDK = True
except ImportError:
    agibot_gdk = None  # type: ignore
    HAS_GDK = False


LEFT_EE_FRAME = "arm_l_end_link"
RIGHT_EE_FRAME = "arm_r_end_link"

GROUP_LEFT_FALLBACK = 4
GROUP_RIGHT_FALLBACK = 8
GROUP_BOTH_FALLBACK = 12

WAIST_JOINTS = [
    "idx01_body_joint1",
    "idx02_body_joint2",
    "idx03_body_joint3",
    "idx04_body_joint4",
    "idx05_body_joint5",
]

LEFT_ARM_JOINTS = [
    "idx21_arm_l_joint1",
    "idx22_arm_l_joint2",
    "idx23_arm_l_joint3",
    "idx24_arm_l_joint4",
    "idx25_arm_l_joint5",
    "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
]

RIGHT_ARM_JOINTS = [
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
]

BOTH_ARM_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS

JOINT_ALIAS_TO_NAME = {
    **{f"lj{i+1}": name for i, name in enumerate(LEFT_ARM_JOINTS)},
    **{f"rj{i+1}": name for i, name in enumerate(RIGHT_ARM_JOINTS)},
}

ARM_JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
    "idx21_arm_l_joint1": (-3.071796, 3.071796),
    "idx22_arm_l_joint2": (-2.059505, 2.059505),
    "idx23_arm_l_joint3": (-3.071796, 3.071796),
    "idx24_arm_l_joint4": (-2.495838, 1.012308),
    "idx25_arm_l_joint5": (-3.071796, 3.071796),
    "idx26_arm_l_joint6": (-1.012308, 1.012308),
    "idx27_arm_l_joint7": (-1.535907, 1.535907),
    "idx61_arm_r_joint1": (-3.071796, 3.071796),
    "idx62_arm_r_joint2": (-2.059505, 2.059505),
    "idx63_arm_r_joint3": (-3.071796, 3.071796),
    "idx64_arm_r_joint4": (-2.495838, 1.012308),
    "idx65_arm_r_joint5": (-3.071796, 3.071796),
    "idx66_arm_r_joint6": (-1.012308, 1.012308),
    "idx67_arm_r_joint7": (-1.535907, 1.535907),
}

WAIST_JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
    "idx01_body_joint1": (-1.082104, 0.000174),
    "idx02_body_joint2": (-0.000174, 2.652900),
    "idx03_body_joint3": (-1.919862, 1.570970),
    "idx04_body_joint4": (-0.436332, 0.436332),
    "idx05_body_joint5": (-3.045599, 3.045599),
}

DEFAULT_POINTS_FILE = "g2_teach_points.json"


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp3(a: Iterable[float], b: Iterable[float], t: float) -> List[float]:
    aa = list(a)
    bb = list(b)
    return [lerp(aa[i], bb[i], t) for i in range(3)]


def rad_to_deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def is_gdk_success(result: Any) -> bool:
    if result is None:
        return True
    name = getattr(result, "name", None)
    if isinstance(name, str) and name == "kSuccess":
        return True
    text = str(result)
    if text in ("kSuccess", "GDKRes.kSuccess"):
        return True
    try:
        return int(result) == 0
    except Exception:
        return False


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    folder = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(folder, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".__g2teach_", suffix=".json", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def quat_norm(q: Iterable[float]) -> List[float]:
    x, y, z, w = list(q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [x / n, y / n, z / n, w / n]


def quat_dot(q0: Iterable[float], q1: Iterable[float]) -> float:
    a = list(q0)
    b = list(q1)
    return sum(x * y for x, y in zip(a, b))


def quat_slerp(q0: Iterable[float], q1: Iterable[float], t: float) -> List[float]:
    a = quat_norm(q0)
    b = quat_norm(q1)
    dot = quat_dot(a, b)
    if dot < 0.0:
        b = [-x for x in b]
        dot = -dot
    dot = clamp(dot, -1.0, 1.0)
    if dot > 0.9995:
        return quat_norm([a[i] + t * (b[i] - a[i]) for i in range(4)])
    theta0 = math.acos(dot)
    sin0 = math.sin(theta0)
    theta = theta0 * t
    sin_t = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_t / sin0
    s1 = sin_t / sin0
    return quat_norm([s0 * a[i] + s1 * b[i] for i in range(4)])


def quat_multiply(q1: Iterable[float], q2: Iterable[float]) -> List[float]:
    x1, y1, z1, w1 = list(q1)
    x2, y2, z2, w2 = list(q2)
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


def quat_from_euler_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> List[float]:
    cr = math.cos(deg_to_rad(roll_deg) * 0.5)
    sr = math.sin(deg_to_rad(roll_deg) * 0.5)
    cp = math.cos(deg_to_rad(pitch_deg) * 0.5)
    sp = math.sin(deg_to_rad(pitch_deg) * 0.5)
    cy = math.cos(deg_to_rad(yaw_deg) * 0.5)
    sy = math.sin(deg_to_rad(yaw_deg) * 0.5)
    return quat_norm([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def quat_angle_deg(q0: Iterable[float], q1: Iterable[float]) -> float:
    a = quat_norm(q0)
    b = quat_norm(q1)
    dot = abs(quat_dot(a, b))
    dot = clamp(dot, -1.0, 1.0)
    return rad_to_deg(2.0 * math.acos(dot))


def sleep_to_rate(next_t: float, dt_s: float) -> float:
    next_t += dt_s
    remain = next_t - time.monotonic()
    if remain > 0:
        time.sleep(remain)
    else:
        next_t = time.monotonic()
    return next_t


def resolve_scope(scope: str) -> str:
    value = scope.strip().lower()
    if value not in {"left", "right", "both"}:
        raise ValueError("scope 必须是 left / right / both")
    return value


def scope_joint_names(scope: str) -> List[str]:
    scope = resolve_scope(scope)
    if scope == "left":
        return LEFT_ARM_JOINTS[:]
    if scope == "right":
        return RIGHT_ARM_JOINTS[:]
    return BOTH_ARM_JOINTS[:]


def pretty_pose(pose: Dict[str, Any]) -> str:
    pos = pose["position"]
    ori = pose["orientation"]
    return (
        f"pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}) m, "
        f"quat=({ori[0]:.4f}, {ori[1]:.4f}, {ori[2]:.4f}, {ori[3]:.4f})"
    )


@dataclass
class G2TeachConfig:
    joint_rate_hz: float = 50.0
    linear_rate_hz: float = 50.0
    joint_lifetime_s: float = 0.10
    linear_lifetime_s: float = 0.02
    hold_final_s: float = 0.20
    default_joint_speed_rad_s: float = 0.35
    default_joint_duration_s: float = 2.0
    linear_max_step_cm: float = 0.5
    linear_max_rot_step_deg: float = 2.0
    min_motion_duration_s: float = 0.20
    joint_jog_duration_s: float = 0.60
    linear_jog_duration_s: float = 0.80
    joint_segment_max_delta_rad: float = 0.35
    joint_goal_tolerance_rad: float = 0.03
    joint_timeout_margin_s: float = 3.0
    joint_poll_hz: float = 20.0
    joint_retry_speed_scale: float = 0.6
    waist_default_speed_rad_s: float = 0.03
    gripper_open_rad: float = 0.0
    gripper_close_rad: float = 0.6


class PointStore:
    def __init__(self, path: str):
        self.path = path
        self.data: Dict[str, Any] = {"version": 1, "points": []}
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.path):
            self.data = {"version": 1, "points": []}
            return
        with open(self.path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict) or "points" not in payload:
            raise RuntimeError(f"点位文件格式无效: {self.path}")
        self.data = payload

    def save(self) -> None:
        atomic_write_json(self.path, self.data)

    def list_points(self) -> List[Dict[str, Any]]:
        return list(self.data.get("points", []))

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        for item in self.list_points():
            if item.get("name") == name:
                return item
        return None

    def upsert(self, point: Dict[str, Any]) -> None:
        points = self.list_points()
        for idx, item in enumerate(points):
            if item.get("name") == point.get("name"):
                points[idx] = point
                self.data["points"] = points
                self.save()
                return
        points.append(point)
        self.data["points"] = points
        self.save()

    def delete(self, name: str) -> bool:
        points = self.list_points()
        new_points = [p for p in points if p.get("name") != name]
        changed = len(new_points) != len(points)
        if changed:
            self.data["points"] = new_points
            self.save()
        return changed


class G2TeachRobot:
    """G2 机器人教学封装类。"""

    def __init__(
        self,
        points_file: str = DEFAULT_POINTS_FILE,
        cfg: Optional[G2TeachConfig] = None,
        simulate: bool = False,
        auto_open: bool = True,
    ):
        self.cfg = cfg or G2TeachConfig()
        self.simulate = simulate
        self.points = PointStore(points_file)

        self._gdk_inited = False
        self._closed = False
        self.robot = None
        self.tf = None
        self._group_left = GROUP_LEFT_FALLBACK
        self._group_right = GROUP_RIGHT_FALLBACK
        self._group_both = GROUP_BOTH_FALLBACK

        self._sim_joint_positions = {name: 0.0 for name in BOTH_ARM_JOINTS + WAIST_JOINTS}
        self._sim_poses = {
            "left": {"position": [0.45, 0.25, 1.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
            "right": {"position": [0.45, -0.25, 1.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
        }
        self._sim_grippers = {"left": 0.0, "right": 0.0}

        if auto_open:
            self.open()

    # ---------- lifecycle ----------

    def open(self) -> None:
        if self.simulate:
            print("[SIM] 以模拟模式启动，不会向机器人发送真实控制命令。")
            return
        if not HAS_GDK:
            raise RuntimeError(
                "未检测到 agibot_gdk。请先 source ~/.cache/agibot/app/env.sh，"
                "或安装 GDK Python 包。"
            )
        result = agibot_gdk.gdk_init()
        if not is_gdk_success(result):
            raise RuntimeError(f"GDK 初始化失败: {result}")
        self._gdk_inited = True
        self.robot = agibot_gdk.Robot()
        self.tf = agibot_gdk.TF()
        time.sleep(1.5)

        grp = getattr(agibot_gdk, "EndEffectorControlGroup", None)
        if grp is not None:
            self._group_left = getattr(grp, "kLeftArm", GROUP_LEFT_FALLBACK)
            self._group_right = getattr(grp, "kRightArm", GROUP_RIGHT_FALLBACK)
            self._group_both = getattr(grp, "kBothArms", GROUP_BOTH_FALLBACK)
        print("[OK] GDK / Robot / TF 初始化完成")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.simulate:
            return
        if self._gdk_inited and HAS_GDK:
            try:
                result = agibot_gdk.gdk_release()
                if is_gdk_success(result):
                    print("[OK] GDK 已释放")
                else:
                    print(f"[WARN] GDK 释放返回: {result}")
            except Exception as exc:
                print(f"[WARN] 释放 GDK 时出错: {exc}")

    def __enter__(self) -> "G2TeachRobot":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---------- state ----------

    def _whole_body_status(self) -> Dict[str, Any]:
        if self.simulate:
            return {
                "left_arm_error": 0,
                "right_arm_error": 0,
                "left_arm_estop": False,
                "right_arm_estop": False,
                "waist_error": 0,
                "chassis_error": 0,
                "neck_error": 0,
            }
        if self.robot is None:
            raise RuntimeError("机器人未初始化")
        return self.robot.get_whole_body_status()

    def _motion_control_status(self) -> Any:
        if self.simulate:
            class SimStatus:
                mode = 5
                error_code = 0
                error_msg = ""
            return SimStatus()
        if self.robot is None:
            raise RuntimeError("机器人未初始化")
        return self.robot.get_motion_control_status()

    def get_mode(self) -> int:
        return int(getattr(self._motion_control_status(), "mode", -1))

    def get_joint_state_map(self) -> Dict[str, Dict[str, Any]]:
        if self.simulate:
            out: Dict[str, Dict[str, Any]] = {}
            for name, pos in self._sim_joint_positions.items():
                out[name] = {
                    "name": name,
                    "position": pos,
                    "motor_position": pos,
                    "velocity": 0.0,
                    "motor_velocity": 0.0,
                    "effort": 0.0,
                    "motor_current": 0.0,
                    "error_code": 0,
                }
            return out
        if self.robot is None:
            raise RuntimeError("机器人未初始化")
        states = self.robot.get_joint_states()
        return {state["name"]: state for state in states["states"]}

    def get_joint_positions(self, joint_names: Iterable[str], use_motor_position: bool = True) -> Dict[str, float]:
        mapping = self.get_joint_state_map()
        key = "motor_position" if use_motor_position else "position"
        result: Dict[str, float] = {}
        for name in joint_names:
            if name not in mapping:
                raise RuntimeError(f"未找到关节状态: {name}")
            result[name] = float(mapping[name][key])
        return result

    def get_arm_joint_positions(self, scope: str = "both") -> Dict[str, float]:
        return self.get_joint_positions(scope_joint_names(scope))

    def get_waist_joint_positions(self) -> Dict[str, float]:
        return self.get_joint_positions(WAIST_JOINTS)

    def get_pose(self, arm: str) -> Dict[str, Any]:
        arm = resolve_scope(arm)
        if arm == "both":
            raise ValueError("get_pose() 仅支持 left/right")
        if self.simulate:
            return copy.deepcopy(self._sim_poses[arm])
        if self.tf is None:
            raise RuntimeError("TF 未初始化")
        frame = LEFT_EE_FRAME if arm == "left" else RIGHT_EE_FRAME
        tf_res = self.tf.get_tf_from_base_link(frame)
        return {
            "position": [
                float(tf_res.translation.x),
                float(tf_res.translation.y),
                float(tf_res.translation.z),
            ],
            "orientation": [
                float(tf_res.rotation.x),
                float(tf_res.rotation.y),
                float(tf_res.rotation.z),
                float(tf_res.rotation.w),
            ],
        }

    def get_both_poses(self) -> Dict[str, Dict[str, Any]]:
        return {"left": self.get_pose("left"), "right": self.get_pose("right")}

    def get_gripper_positions(self) -> Dict[str, float]:
        if self.simulate:
            return copy.deepcopy(self._sim_grippers)
        if self.robot is None:
            raise RuntimeError("机器人未初始化")
        left = 0.0
        right = 0.0
        try:
            es = self.robot.get_end_state()
            left_states = es.get("left_end_state", {}).get("end_states", [])
            right_states = es.get("right_end_state", {}).get("end_states", [])
            if left_states:
                left = float(left_states[0].get("position", 0.0))
            if right_states:
                right = float(right_states[0].get("position", 0.0))
        except Exception:
            pass
        return {"left": left, "right": right}

    def status_dict(self) -> Dict[str, Any]:
        wb = self._whole_body_status()
        mc = self._motion_control_status()
        poses = self.get_both_poses()
        grip = self.get_gripper_positions()
        return {
            "whole_body": wb,
            "motion_control": {
                "mode": int(getattr(mc, "mode", -1)),
                "error_code": int(getattr(mc, "error_code", -1)),
                "error_msg": str(getattr(mc, "error_msg", "")),
            },
            "poses": poses,
            "grippers": grip,
        }

    def format_status(self) -> str:
        st = self.status_dict()
        wb = st["whole_body"]
        mc = st["motion_control"]
        poses = st["poses"]
        grip = st["grippers"]
        return "\n".join([
            "全身状态:",
            f"  left_arm_error={wb.get('left_arm_error', 0)}, right_arm_error={wb.get('right_arm_error', 0)}, "
            f"left_arm_estop={wb.get('left_arm_estop', False)}, right_arm_estop={wb.get('right_arm_estop', False)}",
            f"运控状态: mode={mc['mode']}, error_code={mc['error_code']}, error_msg={mc['error_msg']}",
            f"左臂末端: {pretty_pose(poses['left'])}",
            f"右臂末端: {pretty_pose(poses['right'])}",
            f"夹爪: left={grip['left']:.4f} rad, right={grip['right']:.4f} rad",
        ])

    def ensure_safe_for_motion(self, linear: bool = False) -> None:
        wb = self._whole_body_status()
        errors = []
        if wb.get("left_arm_estop", False):
            errors.append("左臂急停已触发")
        if wb.get("right_arm_estop", False):
            errors.append("右臂急停已触发")
        if int(wb.get("left_arm_error", 0)) != 0:
            errors.append(f"左臂错误码={wb.get('left_arm_error')}")
        if int(wb.get("right_arm_error", 0)) != 0:
            errors.append(f"右臂错误码={wb.get('right_arm_error')}")
        if errors:
            raise RuntimeError("；".join(errors))
        if linear:
            mode = self.get_mode()
            if mode not in (1, 5):
                raise RuntimeError(
                    f"直线运动要求伺服模式 mode=1/5，当前 mode={mode}。请先在示教器/上位机切到伺服模式。"
                )

    # ---------- point store ----------

    def list_points(self, kind: str = "all") -> List[Dict[str, Any]]:
        pts = self.points.list_points()
        kind = kind.lower().strip()
        if kind in {"joint", "linear"}:
            pts = [p for p in pts if p.get("type") == kind]
        return pts

    def get_point(self, name: str) -> Optional[Dict[str, Any]]:
        return self.points.get(name)

    def delete_point(self, name: str) -> bool:
        return self.points.delete(name)

    def reload_points(self) -> None:
        self.points.load()

    def export_points(self, path: str) -> None:
        atomic_write_json(path, self.points.data)

    def capture_joint_point(self, name: str, scope: str = "both", note: str = "") -> Dict[str, Any]:
        scope = resolve_scope(scope)
        point = {
            "name": name,
            "type": "joint",
            "scope": scope,
            "created_at": now_iso(),
            "joint_positions": self.get_arm_joint_positions(scope),
            "waist_positions": self.get_waist_joint_positions(),
            "grippers": self.get_gripper_positions(),
            "note": note,
        }
        return point

    def capture_linear_point(self, name: str, scope: str = "both", note: str = "") -> Dict[str, Any]:
        scope = resolve_scope(scope)
        poses: Dict[str, Any] = {}
        if scope in ("left", "both"):
            poses["left"] = self.get_pose("left")
        if scope in ("right", "both"):
            poses["right"] = self.get_pose("right")
        point = {
            "name": name,
            "type": "linear",
            "scope": scope,
            "created_at": now_iso(),
            "poses": poses,
            "waist_positions": self.get_waist_joint_positions(),
            "grippers": self.get_gripper_positions(),
            "note": note,
        }
        return point

    def save_joint_point(self, name: str, scope: str = "both", note: str = "") -> Dict[str, Any]:
        point = self.capture_joint_point(name, scope=scope, note=note)
        self.points.upsert(point)
        return point

    def save_linear_point(self, name: str, scope: str = "both", note: str = "") -> Dict[str, Any]:
        point = self.capture_linear_point(name, scope=scope, note=note)
        self.points.upsert(point)
        return point

    # ---------- low-level send ----------

    def _wait_until_joint_target(
        self,
        target_positions: Dict[str, float],
        timeout_s: float,
        tolerance_rad: Optional[float] = None,
        poll_hz: Optional[float] = None,
    ) -> Dict[str, float]:
        tol = float(tolerance_rad or self.cfg.joint_goal_tolerance_rad)
        rate = float(poll_hz or self.cfg.joint_poll_hz)
        names = list(target_positions.keys())
        deadline = time.monotonic() + max(0.1, timeout_s)
        last = self.get_joint_positions(names)
        period = 1.0 / max(rate, 1.0)
        while time.monotonic() < deadline:
            last = self.get_joint_positions(names)
            if all(abs(last[n] - target_positions[n]) <= tol for n in names):
                return last
            time.sleep(period)
        detail = ", ".join(
            f"{n}: cur={last[n]:.4f}, tgt={target_positions[n]:.4f}, err={last[n]-target_positions[n]:+.4f}"
            for n in names
        )
        raise RuntimeError(f"关节到位超时: {detail}")

    def _send_joint_command_once(self, positions: Dict[str, float], velocities: Dict[str, float], life_time_s: float) -> None:
        if self.simulate:
            for name, pos in positions.items():
                self._sim_joint_positions[name] = pos
            return
        if self.robot is None:
            raise RuntimeError("机器人未初始化")
        req = agibot_gdk.JointControlReq()
        joint_names = list(positions.keys())
        req.joint_names = joint_names
        req.joint_positions = [float(positions[n]) for n in joint_names]
        req.joint_velocities = [float(velocities[n]) for n in joint_names]
        req.life_time = float(life_time_s)
        req.detail = "g2_teach_robot"
        result = self.robot.joint_control_request(req)
        if not is_gdk_success(result):
            raise RuntimeError(f"joint_control_request 返回异常: {result}")

    def _send_arm_joint_command_once(self, target_positions: Dict[str, float], velocities: Dict[str, float]) -> None:
        if self.simulate:
            for name, pos in target_positions.items():
                self._sim_joint_positions[name] = pos
            return
        if self.robot is None:
            raise RuntimeError("机器人未初始化")
        if not hasattr(self.robot, "move_arm_joint"):
            raise RuntimeError("当前 agibot_gdk 未提供 move_arm_joint")
        current = self.get_joint_positions(BOTH_ARM_JOINTS)
        merged = dict(current)
        merged.update({k: float(v) for k, v in target_positions.items() if k in BOTH_ARM_JOINTS})
        positions = [float(merged[name]) for name in BOTH_ARM_JOINTS]
        vel_map = {name: float(velocities.get(name, self.cfg.default_joint_speed_rad_s)) for name in BOTH_ARM_JOINTS}
        arm_velocities = [max(0.05, vel_map[name]) for name in BOTH_ARM_JOINTS]
        result = self.robot.move_arm_joint(positions, arm_velocities)
        if not is_gdk_success(result):
            raise RuntimeError(f"move_arm_joint 返回异常: {result}")

    def _send_linear_command_once(
        self,
        left_pose: Dict[str, Any],
        right_pose: Dict[str, Any],
        scope: str,
        life_time_s: float,
    ) -> None:
        if self.simulate:
            if scope in ("left", "both"):
                self._sim_poses["left"] = copy.deepcopy(left_pose)
            if scope in ("right", "both"):
                self._sim_poses["right"] = copy.deepcopy(right_pose)
            return
        if self.robot is None:
            raise RuntimeError("机器人未初始化")
        end_pose = agibot_gdk.EndEffectorPose()
        end_pose.life_time = float(life_time_s)
        if scope == "left":
            end_pose.group = int(self._group_left)
        elif scope == "right":
            end_pose.group = int(self._group_right)
        else:
            end_pose.group = int(self._group_both)

        for attr, pose in (("left_end_effector_pose", left_pose), ("right_end_effector_pose", right_pose)):
            obj = getattr(end_pose, attr)
            obj.position.x = float(pose["position"][0])
            obj.position.y = float(pose["position"][1])
            obj.position.z = float(pose["position"][2])
            obj.orientation.x = float(pose["orientation"][0])
            obj.orientation.y = float(pose["orientation"][1])
            obj.orientation.z = float(pose["orientation"][2])
            obj.orientation.w = float(pose["orientation"][3])

        result = self.robot.end_effector_pose_control(end_pose)
        if not is_gdk_success(result):
            raise RuntimeError(f"end_effector_pose_control 返回异常: {result}")

    def _segment_joint_targets(
        self,
        current: Dict[str, float],
        target: Dict[str, float],
        segment_max_delta_rad: float,
    ) -> List[Dict[str, float]]:
        max_delta = max(abs(target[n] - current[n]) for n in target) if target else 0.0
        segments = max(1, int(math.ceil(max_delta / max(segment_max_delta_rad, 1e-6))))
        return [{n: lerp(current[n], target[n], (i + 1) / segments) for n in target} for i in range(segments)]

    # ---------- public motion ----------

    def joint_ptp(
        self,
        target_positions: Dict[str, float],
        duration_s: Optional[float] = None,
        speed_rad_s: Optional[float] = None,
    ) -> None:
        self.ensure_safe_for_motion(linear=False)

        names = list(target_positions.keys())
        if not names:
            return

        current = self.get_joint_positions(names)
        limited_target: Dict[str, float] = {}
        for name, target in target_positions.items():
            if name in ARM_JOINT_LIMITS:
                lo, hi = ARM_JOINT_LIMITS[name]
                clipped = clamp(float(target), lo, hi)
                limited_target[name] = clipped
            else:
                limited_target[name] = float(target)

        max_delta = max(abs(limited_target[n] - current[n]) for n in names)
        if max_delta < self.cfg.joint_goal_tolerance_rad:
            return

        speed = float(speed_rad_s or self.cfg.default_joint_speed_rad_s)
        auto_duration = max(self.cfg.min_motion_duration_s, max_delta / max(speed, 1e-6))
        duration = max(float(duration_s or self.cfg.default_joint_duration_s), auto_duration)

        segments = self._segment_joint_targets(current, limited_target, self.cfg.joint_segment_max_delta_rad)
        use_arm_api = all(name in BOTH_ARM_JOINTS for name in names) and (not self.simulate) and hasattr(self.robot, "move_arm_joint")

        segment_duration = max(self.cfg.min_motion_duration_s, duration / max(1, len(segments)))
        last_start = dict(current)
        for idx, seg_target in enumerate(segments, start=1):
            velocities = {n: max(0.08, abs(seg_target[n] - last_start[n]) / max(segment_duration, 1e-6)) for n in names}
            timeout_s = max(segment_duration + self.cfg.joint_timeout_margin_s, segment_duration * 2.0)
            try:
                if use_arm_api:
                    self._send_arm_joint_command_once(seg_target, velocities)
                else:
                    self._send_joint_command_once(seg_target, velocities, timeout_s)
                self._wait_until_joint_target(seg_target, timeout_s + 1.0)
            except Exception as exc:
                retry_velocities = {n: max(0.05, velocities[n] * self.cfg.joint_retry_speed_scale) for n in names}
                retry_timeout_s = timeout_s + 3.0
                if use_arm_api:
                    self._send_arm_joint_command_once(seg_target, retry_velocities)
                else:
                    self._send_joint_command_once(seg_target, retry_velocities, retry_timeout_s)
                self._wait_until_joint_target(seg_target, retry_timeout_s + 1.0)
            last_start = self.get_joint_positions(names)
        time.sleep(min(self.cfg.hold_final_s, 0.2))

    def linear_ptp(
        self,
        scope: str,
        target_poses: Dict[str, Dict[str, Any]],
        duration_s: Optional[float] = None,
        rate_hz: Optional[float] = None,
        max_step_cm: Optional[float] = None,
        max_rot_step_deg: Optional[float] = None,
    ) -> None:
        scope = resolve_scope(scope)
        self.ensure_safe_for_motion(linear=True)

        rate = float(rate_hz or self.cfg.linear_rate_hz)
        step_cm = float(max_step_cm or self.cfg.linear_max_step_cm)
        rot_step_deg = float(max_rot_step_deg or self.cfg.linear_max_rot_step_deg)

        current = self.get_both_poses()
        goal = {"left": copy.deepcopy(current["left"]), "right": copy.deepcopy(current["right"])}
        if scope in ("left", "both"):
            goal["left"] = copy.deepcopy(target_poses["left"])
        if scope in ("right", "both"):
            goal["right"] = copy.deepcopy(target_poses["right"])

        pos_steps = 1
        rot_steps = 1
        for arm in ("left", "right"):
            if arm not in target_poses:
                continue
            pos_steps = max(pos_steps, int(math.ceil((math.dist(current[arm]["position"], goal[arm]["position"]) * 100.0) / step_cm)))
            rot_steps = max(rot_steps, int(math.ceil(quat_angle_deg(current[arm]["orientation"], goal[arm]["orientation"]) / rot_step_deg)))

        steps = max(pos_steps, rot_steps, 1)
        min_duration = max(self.cfg.min_motion_duration_s, steps / rate)
        duration = max(float(duration_s) if duration_s is not None else min_duration, min_duration)
        dt_s = 1.0 / rate
        life_time_s = max(self.cfg.linear_lifetime_s, 2.0 * dt_s)

        next_t = time.monotonic()
        for i in range(steps):
            t = (i + 1) / steps
            left_pose = {
                "position": lerp3(current["left"]["position"], goal["left"]["position"], t),
                "orientation": quat_slerp(current["left"]["orientation"], goal["left"]["orientation"], t),
            }
            right_pose = {
                "position": lerp3(current["right"]["position"], goal["right"]["position"], t),
                "orientation": quat_slerp(current["right"]["orientation"], goal["right"]["orientation"], t),
            }
            self._send_linear_command_once(left_pose, right_pose, scope, life_time_s)
            next_t = sleep_to_rate(next_t, dt_s)

        hold_steps = max(1, int(math.ceil(self.cfg.hold_final_s * rate)))
        for _ in range(hold_steps):
            self._send_linear_command_once(goal["left"], goal["right"], scope, life_time_s)
            next_t = sleep_to_rate(next_t, dt_s)

    def jog_joint(self, joint_ref: str, delta_rad: float, duration_s: Optional[float] = None) -> None:
        name = JOINT_ALIAS_TO_NAME.get(joint_ref.lower(), joint_ref)
        if name not in ARM_JOINT_LIMITS:
            raise RuntimeError(f"未知关节: {joint_ref}")
        current = self.get_joint_positions([name])[name]
        self.joint_ptp({name: current + float(delta_rad)}, duration_s=duration_s or self.cfg.joint_jog_duration_s)

    def jog_linear(
        self,
        scope: str,
        dx: float,
        dy: float,
        dz: float,
        droll_deg: float = 0.0,
        dpitch_deg: float = 0.0,
        dyaw_deg: float = 0.0,
        duration_s: Optional[float] = None,
    ) -> None:
        scope = resolve_scope(scope)
        current = self.get_both_poses()
        target: Dict[str, Dict[str, Any]] = {}
        delta_q = quat_from_euler_deg(droll_deg, dpitch_deg, dyaw_deg)
        if scope in ("left", "both"):
            left_pose = copy.deepcopy(current["left"])
            left_pose["position"][0] += float(dx)
            left_pose["position"][1] += float(dy)
            left_pose["position"][2] += float(dz)
            left_pose["orientation"] = quat_norm(quat_multiply(left_pose["orientation"], delta_q))
            target["left"] = left_pose
        if scope in ("right", "both"):
            right_pose = copy.deepcopy(current["right"])
            right_pose["position"][0] += float(dx)
            right_pose["position"][1] += float(dy)
            right_pose["position"][2] += float(dz)
            right_pose["orientation"] = quat_norm(quat_multiply(right_pose["orientation"], delta_q))
            target["right"] = right_pose
        self.linear_ptp(scope, target, duration_s=duration_s or self.cfg.linear_jog_duration_s)

    def goto_joint_point(self, name: str, duration_s: Optional[float] = None) -> None:
        point = self.points.get(name)
        if point is None:
            raise RuntimeError(f"未找到点位: {name}")
        if point.get("type") != "joint":
            raise RuntimeError(f"点位 {name} 不是关节点位")
        target = {k: float(v) for k, v in point.get("joint_positions", {}).items()}
        self.joint_ptp(target, duration_s=duration_s)

    def goto_linear_point(self, name: str, duration_s: Optional[float] = None) -> None:
        point = self.points.get(name)
        if point is None:
            raise RuntimeError(f"未找到点位: {name}")
        if point.get("type") != "linear":
            raise RuntimeError(f"点位 {name} 不是直线点位")
        scope = resolve_scope(point.get("scope", "both"))
        poses = copy.deepcopy(point.get("poses", {}))
        self.linear_ptp(scope, poses, duration_s=duration_s)

    def goto_point(self, name: str, duration_s: Optional[float] = None) -> None:
        point = self.points.get(name)
        if point is None:
            raise RuntimeError(f"未找到点位: {name}")
        if point.get("type") == "joint":
            self.goto_joint_point(name, duration_s=duration_s)
        elif point.get("type") == "linear":
            self.goto_linear_point(name, duration_s=duration_s)
        else:
            raise RuntimeError(f"未知点位类型: {point.get('type')}")

    # ---------- gripper ----------

    def set_gripper(self, left: Optional[float] = None, right: Optional[float] = None) -> None:
        current = self.get_gripper_positions()
        tgt_left = current["left"] if left is None else float(left)
        tgt_right = current["right"] if right is None else float(right)
        if self.simulate:
            self._sim_grippers["left"] = tgt_left
            self._sim_grippers["right"] = tgt_right
            return
        if self.robot is None:
            raise RuntimeError("机器人未初始化")
        action = {
            "left_ee_state": {"joint_position": tgt_left},
            "right_ee_state": {"joint_position": tgt_right},
        }
        self.robot.move_ee_pos(action)

    def open_gripper(self, scope: str = "both") -> None:
        scope = resolve_scope(scope)
        if scope == "left":
            self.set_gripper(left=self.cfg.gripper_open_rad)
        elif scope == "right":
            self.set_gripper(right=self.cfg.gripper_open_rad)
        else:
            self.set_gripper(self.cfg.gripper_open_rad, self.cfg.gripper_open_rad)

    def close_gripper(self, scope: str = "both") -> None:
        scope = resolve_scope(scope)
        if scope == "left":
            self.set_gripper(left=self.cfg.gripper_close_rad)
        elif scope == "right":
            self.set_gripper(right=self.cfg.gripper_close_rad)
        else:
            self.set_gripper(self.cfg.gripper_close_rad, self.cfg.gripper_close_rad)

    # ---------- waist ----------

    def move_waist_positions(
        self,
        positions: Iterable[float],
        velocities: Optional[Iterable[float]] = None,
        clip_to_limits: bool = True,
    ) -> List[float]:
        targets = list(float(v) for v in positions)
        if len(targets) != 5:
            raise ValueError("腰部位置必须提供 5 个值")
        if velocities is None:
            vels = [self.cfg.waist_default_speed_rad_s] * 5
        else:
            vels = list(float(v) for v in velocities)
            if len(vels) != 5:
                raise ValueError("腰部速度必须提供 5 个值")

        if clip_to_limits:
            clipped = []
            for name, v in zip(WAIST_JOINTS, targets):
                lo, hi = WAIST_JOINT_LIMITS[name]
                clipped.append(clamp(v, lo, hi))
            targets = clipped

        if self.simulate:
            for name, val in zip(WAIST_JOINTS, targets):
                self._sim_joint_positions[name] = val
            return targets

        if self.robot is None:
            raise RuntimeError("机器人未初始化")
        result = self.robot.move_waist_joint(targets, vels)
        if not is_gdk_success(result):
            raise RuntimeError(f"move_waist_joint 返回异常: {result}")
        return targets

    def move_waist_delta(
        self,
        d1: float = 0.0,
        d2: float = 0.0,
        d3: float = 0.0,
        d4: float = 0.0,
        d5: float = 0.0,
        velocities: Optional[Iterable[float]] = None,
    ) -> List[float]:
        current = self.get_waist_joint_positions()
        target = [
            current["idx01_body_joint1"] + float(d1),
            current["idx02_body_joint2"] + float(d2),
            current["idx03_body_joint3"] + float(d3),
            current["idx04_body_joint4"] + float(d4),
            current["idx05_body_joint5"] + float(d5),
        ]
        return self.move_waist_positions(target, velocities=velocities, clip_to_limits=True)

    def probe_waist_upright(
        self,
        step_rad: float = 0.01,
        keep_idx02: bool = True,
        velocities: Optional[Iterable[float]] = None,
    ) -> List[float]:
        """
        只做一小步“回直”试探：
        - idx01 向 0 靠近
        - idx03 向 0 靠近
        - idx04/idx05 默认保持
        - idx02 默认保持（建议持物时不要动）
        """
        current = self.get_waist_joint_positions()
        target = [
            current["idx01_body_joint1"] + abs(step_rad),
            current["idx02_body_joint2"] if keep_idx02 else current["idx02_body_joint2"],
            current["idx03_body_joint3"] + abs(step_rad),
            current["idx04_body_joint4"],
            current["idx05_body_joint5"],
        ]
        return self.move_waist_positions(target, velocities=velocities, clip_to_limits=True)

    # ---------- compound helpers ----------

    def retract_arms(self, dx: float = -0.10, dz: float = 0.0, duration_s: float = 2.0) -> None:
        self.jog_linear("both", dx=float(dx), dy=0.0, dz=float(dz), duration_s=duration_s)

    def lift_arms(self, dz: float = 0.10, duration_s: float = 2.0) -> None:
        self.jog_linear("both", dx=0.0, dy=0.0, dz=float(dz), duration_s=duration_s)

    def make_carry_posture(
        self,
        retract_x: float = 0.10,
        lift_z: float = 0.0,
        waist_step_rad: float = 0.01,
        waist_max_iters: int = 3,
        arm_duration_s: float = 2.0,
    ) -> Dict[str, Any]:
        """
        推荐的持物运输姿态调整：
        1. 双臂向身体方向后收
        2. 可选上抬
        3. 腰部做若干次小步回直试探；一旦失败即停止
        """
        results: Dict[str, Any] = {"waist_success_steps": 0, "waist_fail_reason": None}
        if abs(retract_x) > 1e-9 or abs(lift_z) > 1e-9:
            self.retract_arms(dx=-abs(retract_x), dz=lift_z, duration_s=arm_duration_s)

        for _ in range(max(0, int(waist_max_iters))):
            try:
                self.probe_waist_upright(step_rad=waist_step_rad)
                results["waist_success_steps"] += 1
                time.sleep(0.3)
            except Exception as exc:
                results["waist_fail_reason"] = str(exc)
                break

        results["status"] = self.status_dict()
        return results

    def move_waist_to_named_joint_point(
        self,
        name: str,
        step_rad: float = 0.01,
        pause_s: float = 0.25,
    ) -> List[float]:
        """
        将腰部逐步移动到某个已保存关节点/直线点里的 waist_positions。
        用小步渐进，尽量降低因重心约束导致的一步失败风险。
        """
        point = self.get_point(name)
        if point is None:
            raise RuntimeError(f"未找到点位: {name}")

        waist_positions = point.get("waist_positions")
        if not waist_positions:
            raise RuntimeError(f"点位 {name} 不含 waist_positions")

        current = self.get_waist_joint_positions()
        current_list = [float(current[jn]) for jn in WAIST_JOINTS]
        target_list = [float(waist_positions[jn]) for jn in WAIST_JOINTS]

        max_delta = max(abs(t - c) for c, t in zip(current_list, target_list))
        if max_delta < 1e-4:
            return target_list

        segments = max(1, int(math.ceil(max_delta / max(step_rad, 1e-6))))
        last = current_list
        for i in range(1, segments + 1):
            alpha = i / segments
            interm = [lerp(c, t, alpha) for c, t in zip(current_list, target_list)]
            try:
                self.move_waist_positions(interm)
            except Exception as exc:
                detail = (
                    f"move_waist_to_named_joint_point({name}) 在第 {i}/{segments} 段失败, "
                    f"目标={interm}, 错误={exc}"
                )
                raise RuntimeError(detail) from exc
            last = interm
            if pause_s > 0:
                time.sleep(pause_s)
        return last

    def goto_named_joint_pose(
        self,
        name: str,
        arm_duration_s: float = 6.0,
        move_waist_first: bool = True,
        waist_step_rad: float = 0.01,
        waist_pause_s: float = 0.25,
    ) -> None:
        """
        执行“腰部 + 双臂”的命名关节点。
        默认顺序:
        1. 腰部先到位
        2. 双臂再到位

        注意:
        - 点位必须是 joint 类型
        - 当前版本的 goto_joint_point() 只执行双臂关节
        """
        point = self.get_point(name)
        if point is None:
            raise RuntimeError(f"未找到点位: {name}")
        if point.get("type") != "joint":
            raise RuntimeError(f"点位 {name} 不是关节点")

        if move_waist_first:
            self.move_waist_to_named_joint_point(name, step_rad=waist_step_rad, pause_s=waist_pause_s)
            self.goto_joint_point(name, duration_s=arm_duration_s)
        else:
            self.goto_joint_point(name, duration_s=arm_duration_s)
            self.move_waist_to_named_joint_point(name, step_rad=waist_step_rad, pause_s=waist_pause_s)

    def run_pick_place(
        self,
        home_name: str = "J_HOME_BOTH",
        grasp_name: str = "J_PICK_BOTH",
        lift_name: str = "L_LIFT_BOTH",
        place_ready_name: str = "J_PLACE_READY_BOTH",
        place_name: str = "L_PLACE_BOTH",
        close_gripper_after_pick: bool = False,
        open_gripper_after_place: bool = False,
        home_duration_s: float = 6.0,
        grasp_duration_s: float = 6.0,
        place_ready_duration_s: float = 4.0,
        linear_duration_s: float = 2.5,
        waist_step_rad: float = 0.01,
        waist_pause_s: float = 0.25,
        return_home: bool = True,
    ) -> Dict[str, Any]:
        """
        跑一条完整的 pick-place 流程:

        1. 回 HOME（腰部 + 双臂）
        2. 到抓取位（腰部先到抓取位置，双臂再到抓取位置）
        3. 可选闭合夹爪
        4. 到抬升点
        5. 到放置预备位
        6. 到放置点
        7. 可选打开夹爪
        8. 可选回 HOME

        返回:
            包含关键阶段状态快照的字典，便于上层日志记录。
        """
        result: Dict[str, Any] = {
            "started_at": now_iso(),
            "home_name": home_name,
            "grasp_name": grasp_name,
            "lift_name": lift_name,
            "place_ready_name": place_ready_name,
            "place_name": place_name,
            "stages": [],
        }

        def snap(stage: str) -> None:
            result["stages"].append({
                "stage": stage,
                "timestamp": now_iso(),
                "status": self.status_dict(),
            })

        # 起步回 HOME
        self.goto_named_joint_pose(
            home_name,
            arm_duration_s=home_duration_s,
            move_waist_first=False,
            waist_step_rad=waist_step_rad,
            waist_pause_s=waist_pause_s,
        )
        snap("home")

        # 到抓取位
        self.goto_named_joint_pose(
            grasp_name,
            arm_duration_s=grasp_duration_s,
            move_waist_first=True,
            waist_step_rad=waist_step_rad,
            waist_pause_s=waist_pause_s,
        )
        snap("grasp")

        if close_gripper_after_pick:
            self.close_gripper("both")
            snap("gripper_closed")

        # 抬升
        if self.get_point(lift_name) is not None:
            self.goto_linear_point(lift_name, duration_s=linear_duration_s)
            snap("lift")

        # 到放置预备位
        if self.get_point(place_ready_name) is not None:
            self.goto_joint_point(place_ready_name, duration_s=place_ready_duration_s)
            snap("place_ready")

        # 到放置点
        if self.get_point(place_name) is not None:
            self.goto_linear_point(place_name, duration_s=linear_duration_s)
            snap("place")

        if open_gripper_after_place:
            self.open_gripper("both")
            snap("gripper_opened")

        if return_home:
            self.goto_named_joint_pose(
                home_name,
                arm_duration_s=home_duration_s,
                move_waist_first=False,
                waist_step_rad=waist_step_rad,
                waist_pause_s=waist_pause_s,
            )
            snap("home_return")

        result["finished_at"] = now_iso()
        result["final_status"] = self.status_dict()
        return result


if __name__ == "__main__":
    print("这个模块提供 G2TeachRobot 类，请在你的工程里 import 使用。")