#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G2 双臂示教器（终端版）

目标：
1. 支持双臂关节点到点运动（PTP / Joint）
2. 支持双臂末端直线点到点运动（LINE / Cartesian）
3. 支持记录点位（关节点位 / 直线点位）
4. 支持手动关节微调和直线微调，用于现场示教
5. 优先保证可用性与稳定性，而不是追求炫技式 UI

设计原则：
- 关节运动：读取当前关节状态，按插值轨迹发送 JointControlReq
- 直线运动：读取当前末端 TF，按直线 + 四元数 SLERP 插值，在 50Hz 下发送 EndEffectorPose
- 数据持久化：点位统一保存到 JSON，采用原子写入，避免中途断电/异常造成文件损坏
- 安全检查：运动前检查急停、错误码、伺服模式、关节限位
- 稳定优先：所有动作都做缓启动/缓结束式插值，不发送阶跃信号

注意：
- end_effector_pose_control 需要机器人处于伺服模式（mode=1/5）
- 直线运动没有碰撞检测，务必在安全环境下使用
- 本程序默认只控制双臂 14 个关节，不控制腰、头、底盘
"""

from __future__ import annotations

import argparse
import cmd
import copy
import datetime as dt
import json
import math
import os
import shlex
import signal
import sys
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


# =========================
# 常量与关节配置
# =========================

LEFT_EE_FRAME = "arm_l_end_link"
RIGHT_EE_FRAME = "arm_r_end_link"

GROUP_LEFT_FALLBACK = 4
GROUP_RIGHT_FALLBACK = 8
GROUP_BOTH_FALLBACK = 12

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

JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
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

JOINT_ALIAS_TO_NAME = {
    **{f"lj{i+1}": name for i, name in enumerate(LEFT_ARM_JOINTS)},
    **{f"rj{i+1}": name for i, name in enumerate(RIGHT_ARM_JOINTS)},
}

DEFAULT_POINTS_FILE = "g2_teach_points.json"


# =========================
# 数学与工具函数
# =========================


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def rad_to_deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")




def is_gdk_success(result: Any) -> bool:
    """兼容 agibot_gdk Python 绑定的多种成功返回形式。"""
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
    fd, tmp_path = tempfile.mkstemp(prefix=".__teach_", suffix=".json", dir=folder)
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


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp3(a: Iterable[float], b: Iterable[float], t: float) -> List[float]:
    a_list = list(a)
    b_list = list(b)
    return [lerp(a_list[i], b_list[i], t) for i in range(3)]


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
        out = [a[i] + t * (b[i] - a[i]) for i in range(4)]
        return quat_norm(out)

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
    d = abs(quat_dot(a, b))
    d = clamp(d, -1.0, 1.0)
    return rad_to_deg(2.0 * math.acos(d))


def sleep_to_rate(next_t: float, period: float) -> float:
    now = time.monotonic()
    if next_t > now:
        time.sleep(next_t - now)
    return next_t + period


def pretty_pose(pose: Dict[str, Any]) -> str:
    pos = pose["position"]
    ori = pose["orientation"]
    return (
        f"pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}) m, "
        f"quat=({ori[0]:.4f}, {ori[1]:.4f}, {ori[2]:.4f}, {ori[3]:.4f})"
    )


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


# =========================
# 点位存储
# =========================


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


# =========================
# 后端接口
# =========================


@dataclass
class TeachConfig:
    joint_rate_hz: float = 50.0
    linear_rate_hz: float = 50.0
    joint_lifetime_s: float = 0.10
    linear_lifetime_s: float = 0.02
    hold_final_s: float = 0.20
    default_joint_speed_rad_s: float = 0.35
    default_joint_duration_s: float = 2.0
    default_linear_duration_s: Optional[float] = None
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
    gripper_open_rad: float = 0.0
    gripper_close_rad: float = 0.6


class G2Backend:
    def __init__(self, simulate: bool = False, cfg: Optional[TeachConfig] = None):
        self.simulate = simulate
        self.cfg = cfg or TeachConfig()
        self._gdk_inited = False
        self._closed = False
        self.robot = None
        self.tf = None
        self._group_left = GROUP_LEFT_FALLBACK
        self._group_right = GROUP_RIGHT_FALLBACK
        self._group_both = GROUP_BOTH_FALLBACK

        self._sim_joint_positions = {name: 0.0 for name in BOTH_ARM_JOINTS}
        self._sim_poses = {
            "left": {
                "position": [0.45, 0.25, 1.00],
                "orientation": [0.0, 0.0, 0.0, 1.0],
            },
            "right": {
                "position": [0.45, -0.25, 1.00],
                "orientation": [0.0, 0.0, 0.0, 1.0],
            },
        }
        self._sim_grippers = {"left": 0.0, "right": 0.0}

    def open(self) -> None:
        if self.simulate:
            print("[SIM] 以模拟模式启动，不会向机器人发送真实控制命令。")
            return
        if not HAS_GDK:
            raise RuntimeError(
                "未检测到 agibot_gdk。请先 source ~/.cache/agibot/app/env.sh，"
                "或使用 --simulate 离线体验。"
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

    # ----- 低层读取 -----

    def _whole_body_status(self) -> Dict[str, Any]:
        if self.simulate:
            return {
                "left_arm_error": 0,
                "right_arm_error": 0,
                "left_arm_estop": False,
                "right_arm_estop": False,
                "chassis_error": 0,
                "waist_error": 0,
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
                frame_names = [LEFT_EE_FRAME, RIGHT_EE_FRAME]
                frame_poses = []
            return SimStatus()
        if self.robot is None:
            raise RuntimeError("机器人未初始化")
        return self.robot.get_motion_control_status()

    def get_joint_state_map(self) -> Dict[str, Dict[str, Any]]:
        if self.simulate:
            state_map: Dict[str, Dict[str, Any]] = {}
            for name, pos in self._sim_joint_positions.items():
                state_map[name] = {
                    "name": name,
                    "position": pos,
                    "motor_position": pos,
                    "velocity": 0.0,
                    "motor_velocity": 0.0,
                    "effort": 0.0,
                    "motor_current": 0.0,
                    "error_code": 0,
                }
            return state_map

        if self.robot is None:
            raise RuntimeError("机器人未初始化")
        states = self.robot.get_joint_states()
        mapping: Dict[str, Dict[str, Any]] = {}
        for state in states["states"]:
            mapping[state["name"]] = state
        return mapping

    def get_joint_positions(self, joint_names: Iterable[str], use_motor_position: bool = True) -> Dict[str, float]:
        mapping = self.get_joint_state_map()
        key = "motor_position" if use_motor_position else "position"
        result: Dict[str, float] = {}
        for name in joint_names:
            if name not in mapping:
                raise RuntimeError(f"未找到关节状态: {name}")
            result[name] = float(mapping[name][key])
        return result

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
        return {
            "left": self.get_pose("left"),
            "right": self.get_pose("right"),
        }

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

    # ----- 状态与安全 -----

    def format_status(self) -> str:
        lines = []
        wb = self._whole_body_status()
        lines.append("全身状态:")
        lines.append(
            f"  left_arm_error={wb.get('left_arm_error', 0)}, right_arm_error={wb.get('right_arm_error', 0)}, "
            f"left_arm_estop={wb.get('left_arm_estop', False)}, right_arm_estop={wb.get('right_arm_estop', False)}"
        )
        try:
            mc = self._motion_control_status()
            lines.append(
                f"运控状态: mode={getattr(mc, 'mode', 'N/A')}, error_code={getattr(mc, 'error_code', 'N/A')}, "
                f"error_msg={getattr(mc, 'error_msg', '')}"
            )
        except Exception as exc:
            lines.append(f"运控状态读取失败: {exc}")

        try:
            poses = self.get_both_poses()
            lines.append(f"左臂末端: {pretty_pose(poses['left'])}")
            lines.append(f"右臂末端: {pretty_pose(poses['right'])}")
        except Exception as exc:
            lines.append(f"末端位姿读取失败: {exc}")

        try:
            grip = self.get_gripper_positions()
            lines.append(f"夹爪: left={grip['left']:.4f} rad, right={grip['right']:.4f} rad")
        except Exception as exc:
            lines.append(f"夹爪状态读取失败: {exc}")
        return "\n".join(lines)

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
            mc = self._motion_control_status()
            mode = int(getattr(mc, "mode", -1))
            if mode not in (1, 5):
                raise RuntimeError(
                    f"直线运动要求伺服模式 mode=1/5，当前 mode={mode}。"
                    "请先在示教器/上位机切到伺服模式。"
                )

    # ----- 点位采集 -----

    def capture_joint_point(self, name: str, scope: str, note: str = "") -> Dict[str, Any]:
        scope = resolve_scope(scope)
        joint_names = scope_joint_names(scope)
        joints = self.get_joint_positions(joint_names)
        point = {
            "name": name,
            "type": "joint",
            "scope": scope,
            "created_at": now_iso(),
            "joint_positions": joints,
            "grippers": self.get_gripper_positions(),
            "note": note,
        }
        return point

    def capture_linear_point(self, name: str, scope: str, note: str = "") -> Dict[str, Any]:
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
            "grippers": self.get_gripper_positions(),
            "note": note,
        }
        return point

    # ----- 低层发送 -----

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
        req.detail = "g2_dual_arm_teach_pendant"
        result = self.robot.joint_control_request(req)
        if not is_gdk_success(result):
            raise RuntimeError(f"joint_control_request 返回异常: {result}")

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

    def set_gripper(self, left: Optional[float] = None, right: Optional[float] = None) -> None:
        current = self.get_gripper_positions()
        tgt_left = current["left"] if left is None else float(left)
        tgt_right = current["right"] if right is None else float(right)
        if self.simulate:
            self._sim_grippers["left"] = tgt_left
            self._sim_grippers["right"] = tgt_right
            print(f"[SIM] 夹爪 -> left={tgt_left:.4f}, right={tgt_right:.4f}")
            return
        if self.robot is None:
            raise RuntimeError("机器人未初始化")
        action = {
            "left_ee_state": {"joint_position": tgt_left},
            "right_ee_state": {"joint_position": tgt_right},
        }
        self.robot.move_ee_pos(action)

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

    def _send_arm_joint_command_once(
        self,
        target_positions: Dict[str, float],
        velocities: Dict[str, float],
    ) -> None:
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

    def _segment_joint_targets(
        self,
        current: Dict[str, float],
        target: Dict[str, float],
        segment_max_delta_rad: float,
    ) -> List[Dict[str, float]]:
        max_delta = max(abs(target[n] - current[n]) for n in target) if target else 0.0
        segments = max(1, int(math.ceil(max_delta / max(segment_max_delta_rad, 1e-6))))
        return [
            {n: lerp(current[n], target[n], (i + 1) / segments) for n in target}
            for i in range(segments)
        ]

    # ----- 高层动作：关节 PTP / 直线 LINE -----

    def joint_ptp(
        self,
        target_positions: Dict[str, float],
        duration_s: Optional[float] = None,
        speed_rad_s: Optional[float] = None,
        rate_hz: Optional[float] = None,
    ) -> None:
        self.ensure_safe_for_motion(linear=False)

        names = list(target_positions.keys())
        if not names:
            return

        current = self.get_joint_positions(names)
        limited_target: Dict[str, float] = {}
        for name, target in target_positions.items():
            if name in JOINT_LIMITS:
                lo, hi = JOINT_LIMITS[name]
                clipped = clamp(float(target), lo, hi)
                if abs(clipped - float(target)) > 1e-9:
                    print(f"[WARN] {name} 超限，已裁剪到 {clipped:.6f} rad")
                limited_target[name] = clipped
            else:
                limited_target[name] = float(target)

        max_delta = max(abs(limited_target[n] - current[n]) for n in names)
        if max_delta < self.cfg.joint_goal_tolerance_rad:
            print("[MOVE][JOINT] 已在目标附近，无需移动")
            return

        speed = float(speed_rad_s or self.cfg.default_joint_speed_rad_s)
        auto_duration = max(self.cfg.min_motion_duration_s, max_delta / max(speed, 1e-6))
        duration = max(float(duration_s or self.cfg.default_joint_duration_s), auto_duration)

        segments = self._segment_joint_targets(current, limited_target, self.cfg.joint_segment_max_delta_rad)
        use_arm_api = all(name in BOTH_ARM_JOINTS for name in names) and (not self.simulate) and hasattr(self.robot, "move_arm_joint")
        print(
            f"[MOVE][JOINT] duration={duration:.3f}s, max_delta={max_delta:.4f} rad, "
            f"segments={len(segments)}, api={'move_arm_joint' if use_arm_api else 'joint_control_request'}"
        )

        segment_duration = max(self.cfg.min_motion_duration_s, duration / max(1, len(segments)))
        last_start = dict(current)
        for idx, seg_target in enumerate(segments, start=1):
            seg_delta = max(abs(seg_target[n] - last_start[n]) for n in names)
            seg_speed = max(0.08, seg_delta / max(segment_duration, 1e-6))
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
                print(f"[WARN] 关节段 {idx}/{len(segments)} 失败，准备慢速重试: {exc}")
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
        if rate <= 0:
            raise ValueError("linear_rate_hz 必须大于 0")
        step_cm = float(max_step_cm or self.cfg.linear_max_step_cm)
        rot_step_deg = float(max_rot_step_deg or self.cfg.linear_max_rot_step_deg)
        if step_cm <= 0 or rot_step_deg <= 0:
            raise ValueError("linear_max_step_cm / linear_max_rot_step_deg 必须大于 0")

        current = self.get_both_poses()
        goal = {
            "left": copy.deepcopy(current["left"]),
            "right": copy.deepcopy(current["right"]),
        }
        if scope in ("left", "both"):
            goal["left"] = copy.deepcopy(target_poses["left"])
        if scope in ("right", "both"):
            goal["right"] = copy.deepcopy(target_poses["right"])

        pos_steps = 1
        rot_steps = 1
        for arm in ("left", "right"):
            if arm not in target_poses:
                continue
            cur_p = current[arm]["position"]
            tgt_p = goal[arm]["position"]
            dist_m = math.dist(cur_p, tgt_p)
            pos_steps = max(pos_steps, int(math.ceil((dist_m * 100.0) / step_cm)))

            cur_q = current[arm]["orientation"]
            tgt_q = goal[arm]["orientation"]
            angle_deg = quat_angle_deg(cur_q, tgt_q)
            rot_steps = max(rot_steps, int(math.ceil(angle_deg / rot_step_deg)))

        steps = max(pos_steps, rot_steps, 1)
        min_duration = max(self.cfg.min_motion_duration_s, steps / rate)
        duration = max(float(duration_s) if duration_s is not None else min_duration, min_duration)
        dt_s = 1.0 / rate
        life_time_s = max(self.cfg.linear_lifetime_s, 2.0 * dt_s)

        print(
            f"[MOVE][LINE] scope={scope}, duration={duration:.3f}s, steps={steps}, "
            f"pos_steps={pos_steps}, rot_steps={rot_steps}"
        )

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

    # ----- 示教辅助：jog -----

    def jog_joint(self, joint_ref: str, delta_rad: float, duration_s: Optional[float] = None) -> None:
        name = JOINT_ALIAS_TO_NAME.get(joint_ref.lower(), joint_ref)
        if name not in JOINT_LIMITS:
            raise RuntimeError(f"未知关节: {joint_ref}")
        current = self.get_joint_positions([name])[name]
        target = current + float(delta_rad)
        self.joint_ptp({name: target}, duration_s=duration_s or self.cfg.joint_jog_duration_s)

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


# =========================
# 交互式示教器
# =========================


class TeachPendantShell(cmd.Cmd):
    intro = (
        "\n"
        "G2 双臂示教器已启动。输入 help 查看命令。\n"
        "建议先执行 status，再根据需要 savej / savel / goto / jogj / jogl。\n"
    )
    prompt = "g2-teach> "

    def __init__(self, backend: G2Backend, store: PointStore):
        super().__init__()
        self.backend = backend
        self.store = store

    def emptyline(self) -> bool:
        return False

    def default(self, line: str) -> None:
        print(f"未知命令: {line!r}。输入 help 查看可用命令。")

    def _split(self, arg: str) -> List[str]:
        return shlex.split(arg)

    def _print_point(self, point: Dict[str, Any]) -> None:
        print(f"name={point['name']}")
        print(f"type={point['type']}, scope={point['scope']}, created_at={point.get('created_at', '')}")
        note = point.get("note", "")
        if note:
            print(f"note={note}")
        if point["type"] == "joint":
            for name, pos in point.get("joint_positions", {}).items():
                print(f"  {name}: {pos:.6f} rad ({rad_to_deg(pos):.2f} deg)")
        elif point["type"] == "linear":
            for arm, pose in point.get("poses", {}).items():
                print(f"  {arm}: {pretty_pose(pose)}")
        grip = point.get("grippers")
        if isinstance(grip, dict):
            print(f"grippers: left={grip.get('left', 0.0):.4f}, right={grip.get('right', 0.0):.4f}")

    # ----- 状态 -----

    def do_status(self, arg: str) -> None:
        """status
        查看机器人全身状态、运控模式、双臂末端位姿和夹爪位置。
        """
        print(self.backend.format_status())

    def do_list(self, arg: str) -> None:
        """list [all|joint|linear]
        列出已保存点位。
        示例:
          list
          list joint
          list linear
        """
        kind = (arg.strip().lower() or "all")
        points = self.store.list_points()
        if kind != "all":
            points = [p for p in points if p.get("type") == kind]
        if not points:
            print("暂无点位。")
            return
        for p in points:
            print(f"- {p['name']:<24} type={p['type']:<6} scope={p['scope']:<5} created_at={p.get('created_at','')}")

    def do_show(self, arg: str) -> None:
        """show <name>
        查看单个点位详情。
        """
        name = arg.strip()
        if not name:
            print("用法: show <name>")
            return
        point = self.store.get(name)
        if point is None:
            print(f"未找到点位: {name}")
            return
        self._print_point(point)

    # ----- 记录点位 -----

    def do_savej(self, arg: str) -> None:
        """savej <name> [left|right|both] [note...]
        保存当前关节点位。
        示例:
          savej home both
          savej left_ready left 左臂预备位
        """
        parts = self._split(arg)
        if len(parts) < 1:
            print("用法: savej <name> [left|right|both] [note...]")
            return
        name = parts[0]
        scope = "both"
        note = ""
        if len(parts) >= 2 and parts[1].lower() in {"left", "right", "both"}:
            scope = parts[1].lower()
            note = " ".join(parts[2:])
        else:
            note = " ".join(parts[1:])
        point = self.backend.capture_joint_point(name, scope, note)
        self.store.upsert(point)
        print(f"[OK] 已保存关节点位: {name} (scope={scope})")

    def do_savel(self, arg: str) -> None:
        """savel <name> [left|right|both] [note...]
        保存当前直线点位（末端位姿点）。
        示例:
          savel pick_left left
          savel sync_pose both 双臂同步姿态
        """
        parts = self._split(arg)
        if len(parts) < 1:
            print("用法: savel <name> [left|right|both] [note...]")
            return
        name = parts[0]
        scope = "both"
        note = ""
        if len(parts) >= 2 and parts[1].lower() in {"left", "right", "both"}:
            scope = parts[1].lower()
            note = " ".join(parts[2:])
        else:
            note = " ".join(parts[1:])
        point = self.backend.capture_linear_point(name, scope, note)
        self.store.upsert(point)
        print(f"[OK] 已保存直线点位: {name} (scope={scope})")

    # ----- 执行点位 -----

    def do_goto(self, arg: str) -> None:
        """goto <name> [duration_s]
        自动识别点位类型并执行。
        - joint 点：执行关节 PTP
        - linear 点：执行末端 LINE
        """
        parts = self._split(arg)
        if not parts:
            print("用法: goto <name> [duration_s]")
            return
        name = parts[0]
        duration = float(parts[1]) if len(parts) >= 2 else None
        point = self.store.get(name)
        if point is None:
            print(f"未找到点位: {name}")
            return
        self._execute_point(point, duration)

    def do_gotoj(self, arg: str) -> None:
        """gotoj <name> [duration_s]
        仅执行关节点位。
        """
        parts = self._split(arg)
        if not parts:
            print("用法: gotoj <name> [duration_s]")
            return
        name = parts[0]
        duration = float(parts[1]) if len(parts) >= 2 else None
        point = self.store.get(name)
        if point is None or point.get("type") != "joint":
            print(f"未找到关节点位: {name}")
            return
        self._execute_point(point, duration)

    def do_gotol(self, arg: str) -> None:
        """gotol <name> [duration_s] [max_step_cm]
        仅执行直线点位。
        """
        parts = self._split(arg)
        if not parts:
            print("用法: gotol <name> [duration_s] [max_step_cm]")
            return
        name = parts[0]
        duration = float(parts[1]) if len(parts) >= 2 else None
        step_cm = float(parts[2]) if len(parts) >= 3 else None
        point = self.store.get(name)
        if point is None or point.get("type") != "linear":
            print(f"未找到直线点位: {name}")
            return
        self._execute_point(point, duration, step_cm)

    def _execute_point(self, point: Dict[str, Any], duration_s: Optional[float], max_step_cm: Optional[float] = None) -> None:
        if point["type"] == "joint":
            joints = point.get("joint_positions", {})
            print(f"[RUN] 执行关节点位 {point['name']} (scope={point['scope']})")
            self.backend.joint_ptp(joints, duration_s=duration_s)
        elif point["type"] == "linear":
            poses = point.get("poses", {})
            print(f"[RUN] 执行直线点位 {point['name']} (scope={point['scope']})")
            self.backend.linear_ptp(point["scope"], poses, duration_s=duration_s, max_step_cm=max_step_cm)
        else:
            raise RuntimeError(f"未知点位类型: {point['type']}")
        print("[OK] 动作执行完成")

    # ----- 手动示教 -----

    def do_jogj(self, arg: str) -> None:
        """jogj <joint_alias|joint_name> <delta_rad> [duration_s]
        关节微调。
        支持别名: lj1..lj7 / rj1..rj7
        示例:
          jogj lj3 0.10
          jogj rj5 -0.05 0.8
          jogj idx21_arm_l_joint1 0.2
        """
        parts = self._split(arg)
        if len(parts) < 2:
            print("用法: jogj <joint_alias|joint_name> <delta_rad> [duration_s]")
            return
        joint_ref = parts[0]
        delta_rad = float(parts[1])
        duration = float(parts[2]) if len(parts) >= 3 else None
        self.backend.jog_joint(joint_ref, delta_rad, duration)
        print("[OK] 关节微调完成")

    def do_jogl(self, arg: str) -> None:
        """jogl <left|right|both> <dx> <dy> <dz> [droll_deg dpitch_deg dyaw_deg] [duration_s]
        末端直线微调。
        位置单位 m，姿态单位 deg。
        示例:
          jogl left 0.02 0 0
          jogl right 0 0.01 -0.02 0 0 5
          jogl both 0 0 0.03 0 0 0 1.2
        """
        parts = self._split(arg)
        if len(parts) < 4:
            print("用法: jogl <left|right|both> <dx> <dy> <dz> [droll_deg dpitch_deg dyaw_deg] [duration_s]")
            return
        scope = parts[0]
        dx = float(parts[1])
        dy = float(parts[2])
        dz = float(parts[3])
        droll = float(parts[4]) if len(parts) >= 5 else 0.0
        dpitch = float(parts[5]) if len(parts) >= 6 else 0.0
        dyaw = float(parts[6]) if len(parts) >= 7 else 0.0
        duration = float(parts[7]) if len(parts) >= 8 else None
        self.backend.jog_linear(scope, dx, dy, dz, droll, dpitch, dyaw, duration)
        print("[OK] 直线微调完成")

    def do_gripper(self, arg: str) -> None:
        """gripper [left_pos] [right_pos]
        设置夹爪位置（弧度）。
        示例:
          gripper 0.0 0.0
          gripper 0.6
        """
        parts = self._split(arg)
        if not parts:
            grip = self.backend.get_gripper_positions()
            print(f"left={grip['left']:.4f}, right={grip['right']:.4f}")
            return
        left = float(parts[0]) if len(parts) >= 1 else None
        right = float(parts[1]) if len(parts) >= 2 else None
        self.backend.set_gripper(left, right)
        print("[OK] 夹爪指令已发送")

    def do_open(self, arg: str) -> None:
        """open [left|right|both]
        打开夹爪。
        """
        scope = arg.strip().lower() or "both"
        if scope not in {"left", "right", "both"}:
            print("用法: open [left|right|both]")
            return
        if scope == "left":
            self.backend.set_gripper(left=self.backend.cfg.gripper_open_rad)
        elif scope == "right":
            self.backend.set_gripper(right=self.backend.cfg.gripper_open_rad)
        else:
            self.backend.set_gripper(
                left=self.backend.cfg.gripper_open_rad,
                right=self.backend.cfg.gripper_open_rad,
            )
        print("[OK] 打开夹爪指令已发送")

    def do_close(self, arg: str) -> None:
        """close [left|right|both]
        关闭夹爪。
        """
        scope = arg.strip().lower() or "both"
        if scope not in {"left", "right", "both"}:
            print("用法: close [left|right|both]")
            return
        if scope == "left":
            self.backend.set_gripper(left=self.backend.cfg.gripper_close_rad)
        elif scope == "right":
            self.backend.set_gripper(right=self.backend.cfg.gripper_close_rad)
        else:
            self.backend.set_gripper(
                left=self.backend.cfg.gripper_close_rad,
                right=self.backend.cfg.gripper_close_rad,
            )
        print("[OK] 关闭夹爪指令已发送")

    # ----- 点位管理 -----

    def do_delete(self, arg: str) -> None:
        """delete <name>
        删除点位。
        """
        name = arg.strip()
        if not name:
            print("用法: delete <name>")
            return
        if self.store.delete(name):
            print(f"[OK] 已删除点位: {name}")
        else:
            print(f"未找到点位: {name}")

    def do_export(self, arg: str) -> None:
        """export [path]
        导出当前点位 JSON 到指定路径；不指定则打印当前文件路径。
        """
        target = arg.strip()
        if not target:
            print(os.path.abspath(self.store.path))
            return
        atomic_write_json(target, self.store.data)
        print(f"[OK] 已导出到: {target}")

    def do_reload(self, arg: str) -> None:
        """reload
        重新加载点位文件。
        """
        self.store.load()
        print("[OK] 点位文件已重新加载")

    # ----- 退出 -----

    def do_exit(self, arg: str) -> bool:
        """exit
        退出示教器。
        """
        return True

    def do_quit(self, arg: str) -> bool:
        """quit
        退出示教器。
        """
        return True

    def do_EOF(self, arg: str) -> bool:
        print()
        return True


# =========================
# 启动入口
# =========================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G2 双臂示教器（终端版）")
    parser.add_argument(
        "--points",
        default=DEFAULT_POINTS_FILE,
        help=f"点位文件路径，默认: {DEFAULT_POINTS_FILE}",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="模拟模式：不连接真实机器人，可离线测试点位管理和命令交互",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backend = G2Backend(simulate=args.simulate)
    store = PointStore(args.points)

    def _cleanup(*_args: Any) -> None:
        backend.close()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    try:
        backend.open()
        shell = TeachPendantShell(backend, store)
        shell.cmdloop()
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[FATAL] {exc}")
        return 1
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())