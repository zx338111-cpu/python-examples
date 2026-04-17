#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G2RobotController —— 智元 G2 机器人高级控制封装（完整整合版）

说明
----
1. 统一保留了腰部、头部、左臂、右臂、双臂、双臂末端、夹爪、全身点位等接口。
2. 点位会自动持久化到脚本同目录 g2_points/ 下，重启后自动恢复。
3. 关节控制优先尝试 SDK 专用接口；失败时自动回退到 JointControlReq 方案。
4. 末端伺服使用 50Hz 插值，左右臂通过同一个 EndEffectorPose 同步发送。
"""

import datetime
import json
import math
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

try:
    import agibot_gdk
    _GDK_AVAILABLE = True
except ImportError:
    print("⚠️  未找到 agibot_gdk，进入模拟模式")
    agibot_gdk = None
    _GDK_AVAILABLE = False


# ============================================================================
# 常量
# ============================================================================

JOINT_NAMES: List[str] = [
    "idx01_body_joint1", "idx02_body_joint2", "idx03_body_joint3",
    "idx04_body_joint4", "idx05_body_joint5",
    "idx11_head_joint1", "idx12_head_joint2", "idx13_head_joint3",
    "idx21_arm_l_joint1", "idx22_arm_l_joint2", "idx23_arm_l_joint3",
    "idx24_arm_l_joint4", "idx25_arm_l_joint5", "idx26_arm_l_joint6", "idx27_arm_l_joint7",
    "idx61_arm_r_joint1", "idx62_arm_r_joint2", "idx63_arm_r_joint3",
    "idx64_arm_r_joint4", "idx65_arm_r_joint5", "idx66_arm_r_joint6", "idx67_arm_r_joint7",
]

JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
    "idx01_body_joint1": (-1.082104,  0.000174),
    "idx02_body_joint2": (-0.000174,  2.652900),
    "idx03_body_joint3": (-1.919862,  1.570970),
    "idx04_body_joint4": (-0.436332,  0.436332),
    "idx05_body_joint5": (-3.045599,  3.045599),
    "idx11_head_joint1": (-1.570970,  1.570970),
    "idx12_head_joint2": (-0.349240,  0.349240),
    "idx13_head_joint3": (-0.534773,  0.534773),
    "idx21_arm_l_joint1": (-3.071796,  3.071796),
    "idx22_arm_l_joint2": (-2.059505,  2.059505),
    "idx23_arm_l_joint3": (-3.071796,  3.071796),
    "idx24_arm_l_joint4": (-2.495838,  1.012308),
    "idx25_arm_l_joint5": (-3.071796,  3.071796),
    "idx26_arm_l_joint6": (-1.012308,  1.012308),
    "idx27_arm_l_joint7": (-1.535907,  1.535907),
    "idx61_arm_r_joint1": (-3.071796,  3.071796),
    "idx62_arm_r_joint2": (-2.059505,  2.059505),
    "idx63_arm_r_joint3": (-3.071796,  3.071796),
    "idx64_arm_r_joint4": (-2.495838,  1.012308),
    "idx65_arm_r_joint5": (-3.071796,  3.071796),
    "idx66_arm_r_joint6": (-1.012308,  1.012308),
    "idx67_arm_r_joint7": (-1.535907,  1.535907),
}

WAIST_JOINT_NAMES = JOINT_NAMES[0:5]
HEAD_JOINT_NAMES = JOINT_NAMES[5:8]
LEFT_ARM_JOINT_NAMES = JOINT_NAMES[8:15]
RIGHT_ARM_JOINT_NAMES = JOINT_NAMES[15:22]

LEFT_EE_FRAME = "arm_l_end_link"
RIGHT_EE_FRAME = "arm_r_end_link"

SERVO_HZ = 50.0
SERVO_PERIOD = 1.0 / SERVO_HZ


# ============================================================================
# 工具函数
# ============================================================================

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _clamp_joint(name: str, pos: float) -> float:
    if name in JOINT_LIMITS:
        lo, hi = JOINT_LIMITS[name]
        c = _clamp(pos, lo, hi)
        if abs(c - pos) > 1e-6:
            print(f"  ⚠️  限位保护: {name}  {pos:.4f}→{c:.4f}  [{lo:.4f},{hi:.4f}]")
        return c
    return pos


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp3(a: List[float], b: List[float], t: float) -> List[float]:
    return [_lerp(a[i], b[i], t) for i in range(3)]


def _quat_norm(q: List[float]) -> List[float]:
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    return [0.0, 0.0, 0.0, 1.0] if n < 1e-12 else [x / n, y / n, z / n, w / n]


def _quat_slerp(q0: List[float], q1: List[float], t: float) -> List[float]:
    q0 = _quat_norm(q0)
    q1 = list(_quat_norm(q1))
    dot = sum(a * b for a, b in zip(q0, q1))
    if dot < 0:
        q1 = [-x for x in q1]
        dot = -dot
    dot = _clamp(dot, -1.0, 1.0)
    if dot > 0.9995:
        return _quat_norm([q0[i] + t * (q1[i] - q0[i]) for i in range(4)])
    th0 = math.acos(dot)
    s0 = math.sin(th0)
    th = th0 * t
    a = math.cos(th) - dot * math.sin(th) / s0
    b = math.sin(th) / s0
    return _quat_norm([a * q0[i] + b * q1[i] for i in range(4)])


def _precise_sleep(next_t: float, period: float) -> float:
    now = time.monotonic()
    if next_t > now:
        time.sleep(next_t - now)
    return next_t + period


def _load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)









# ============================================================================
# 持久化点位库
# ============================================================================

class _PointStore:
    def __init__(self, filepath: str, label: str):
        self._path = filepath
        self._label = label
        self._data: Dict[str, list] = {}
        self._load()



    def clamp_xyz_offset(
        self,
        dx: float,
        dy: float,
        dz: float,
        x_limit: float = 0.04,
        y_limit: float = 0.04,
        z_limit: float = 0.02,
    ):
        dx = max(-x_limit, min(x_limit, dx))
        dy = max(-y_limit, min(y_limit, dy))
        dz = max(-z_limit, min(z_limit, dz))
        return dx, dy, dz







    def _load(self) -> None:
        raw = _load_json(self._path)
        pts = raw.get("points", raw)
        self._data = {k: v for k, v in pts.items() if isinstance(v, list)}
        if self._data:
            print(f"  📂 [{self._label}] 从磁盘恢复 {len(self._data)} 个点位: {list(self._data.keys())}")

    def _flush(self) -> None:
        _save_json(
            self._path,
            {
                "meta": {
                    "updated": datetime.datetime.now().isoformat(),
                    "label": self._label,
                },
                "points": self._data,
            },
        )

    def set(self, name: str, positions: list) -> None:
        self._data[name] = [float(p) for p in positions]
        self._flush()
        print(f"  💾 [{self._label}] 点位 '{name}' 已保存到磁盘")

    def get(self, name: str) -> Optional[list]:
        return self._data.get(name)

    def delete(self, name: str) -> bool:
        if name not in self._data:
            print(f"  ⚠️  [{self._label}] 点位 '{name}' 不存在，可用: {list(self._data.keys())}")
            return False
        del self._data[name]
        self._flush()
        print(f"  🗑️  [{self._label}] 点位 '{name}' 已删除并同步磁盘")
        return True

    def list_all(self) -> None:
        if not self._data:
            print(f"  （[{self._label}] 暂无点位）")
            return
        print(f"\n  [{self._label}] 共 {len(self._data)} 个点位：")
        for n, v in self._data.items():
            if len(v) in (3, 4, 7):
                vals = "  ".join(f"{math.degrees(p):+7.2f}°" for p in v)
                print(f"    {n:20s}  [{vals}]")
            else:
                print(f"    {n:20s}  len={len(v)}")

    def keys(self) -> List[str]:
        return list(self._data.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._data


# ============================================================================
# 主控制类
# ============================================================================

class G2RobotController:
    def __init__(self, points_dir: Optional[str] = None):
        if points_dir is None:
            points_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "g2_points")
        os.makedirs(points_dir, exist_ok=True)

        self._ws = _PointStore(os.path.join(points_dir, "waist_points.json"), "腰部")
        self._la = _PointStore(os.path.join(points_dir, "left_arm_points.json"), "左臂")
        self._ra = _PointStore(os.path.join(points_dir, "right_arm_points.json"), "右臂")
        self._bd = _PointStore(os.path.join(points_dir, "body_points.json"), "全身")

        self.robot = None
        self.tf = None
        self._gdk_inited = False
        self._lock = threading.RLock()

        # 尽量兼容不同 SDK 枚举值
        self._grp_left = 4
        self._grp_right = 8
        self._grp_both = 12

    # ------------------------------------------------------------------
    # 初始化 / 释放
    # ------------------------------------------------------------------

    def init(self) -> bool:
        if not _GDK_AVAILABLE:
            print("⚠️  GDK 不可用，模拟模式")
            return False

        print("🔧 初始化 GDK...")
        if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
            print("❌ GDK 初始化失败")
            return False

        self.robot = agibot_gdk.Robot()
        time.sleep(2.0)

        try:
            self.tf = agibot_gdk.TF()
            time.sleep(0.5)
        except Exception as e:
            self.tf = None
            print(f"⚠️  TF 初始化失败（末端伺服不可用）: {e}")

        try:
            grp = agibot_gdk.EndEffectorControlGroup
            for left_name in ("kLeft", "kLeftArm"):
                if hasattr(grp, left_name):
                    self._grp_left = int(getattr(grp, left_name))
                    break
            for right_name in ("kRight", "kRightArm"):
                if hasattr(grp, right_name):
                    self._grp_right = int(getattr(grp, right_name))
                    break
            for both_name in ("kBoth", "kBothArms"):
                if hasattr(grp, both_name):
                    self._grp_both = int(getattr(grp, both_name))
                    break
        except Exception:
            pass

        self._gdk_inited = True
        print("✅ GDK 初始化完成")
        return True

    def release(self) -> None:
        if not self._gdk_inited or not _GDK_AVAILABLE:
            return
        try:
            agibot_gdk.gdk_release()
        finally:
            self._gdk_inited = False
            print("✅ GDK 已释放")

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _check_init(self) -> bool:
        if not self._gdk_inited or self.robot is None:
            print("❌ 未初始化，请先调用 ctrl.init()")
            return False
        return True

    def _send_joint_cmd(
        self,
        names: List[str],
        positions: List[float],
        velocity: float = 0.3,
        life_time: float = 1.0,
    ) -> None:
        safe = [_clamp_joint(n, p) for n, p in zip(names, positions)]
        req = agibot_gdk.JointControlReq()
        req.life_time = float(life_time)
        req.joint_names = list(names)
        req.joint_positions = list(safe)
        req.joint_velocities = [float(velocity)]
        self.robot.joint_control_request(req)

    def _get_joint_map(self) -> Dict[str, float]:
        s = self.robot.get_joint_states()
        return {x["name"]: x["motor_position"] for x in s["states"]}

    def _get_ee_tf(self, frame: str) -> Tuple[List[float], List[float]]:
        t = self.tf.get_tf_from_base_link(frame)
        return (
            [t.translation.x, t.translation.y, t.translation.z],
            [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w],
        )

    def _send_ee_cmd(
        self,
        l_pos: List[float],
        l_quat: List[float],
        r_pos: List[float],
        r_quat: List[float],
        life_time: float = SERVO_PERIOD * 2,
    ) -> None:
        ep = agibot_gdk.EndEffectorPose()
        ep.life_time = float(life_time)
        ep.group = int(self._grp_both)

        ep.left_end_effector_pose.position.x = float(l_pos[0])
        ep.left_end_effector_pose.position.y = float(l_pos[1])
        ep.left_end_effector_pose.position.z = float(l_pos[2])
        ep.left_end_effector_pose.orientation.x = float(l_quat[0])
        ep.left_end_effector_pose.orientation.y = float(l_quat[1])
        ep.left_end_effector_pose.orientation.z = float(l_quat[2])
        ep.left_end_effector_pose.orientation.w = float(l_quat[3])

        ep.right_end_effector_pose.position.x = float(r_pos[0])
        ep.right_end_effector_pose.position.y = float(r_pos[1])
        ep.right_end_effector_pose.position.z = float(r_pos[2])
        ep.right_end_effector_pose.orientation.x = float(r_quat[0])
        ep.right_end_effector_pose.orientation.y = float(r_quat[1])
        ep.right_end_effector_pose.orientation.z = float(r_quat[2])
        ep.right_end_effector_pose.orientation.w = float(r_quat[3])

        self.robot.end_effector_pose_control(ep)

    def _ensure_servo_mode(self, timeout: float = 5.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                mode = self.robot.get_motion_control_status().mode
                if mode in (1, 5):
                    return True
                self.robot.switch_motion_control_mode(5)
            except Exception as e:
                print(f"  ⚠️  切换伺服模式: {e}")
            time.sleep(0.3)
        print("❌ 无法进入伺服模式")
        return False

    def _wait_arrive(self, target: Dict[str, float], tol: float = 0.05, timeout: float = 15.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                cur = self._get_joint_map()
                if all(abs(cur.get(n, 0.0) - p) < tol for n, p in target.items()):
                    return True
            except Exception:
                pass
            time.sleep(0.05)
        print(f"  ⚠️  等待到位超时（{timeout:.0f}s）")
        return False

    def _servo_move(self, arm: str, target_pos: List[float], target_quat: Optional[List[float]], duration_s: float) -> bool:
        if self.tf is None:
            print("❌ TF 不可用")
            return False
        if not self._ensure_servo_mode():
            return False

        l_sp, l_sq = self._get_ee_tf(LEFT_EE_FRAME)
        r_sp, r_sq = self._get_ee_tf(RIGHT_EE_FRAME)

        if arm == "left":
            sp, sq = l_sp, l_sq
        else:
            sp, sq = r_sp, r_sq
        eq = _quat_norm(target_quat) if target_quat else sq

        steps = max(int(duration_s * SERVO_HZ), 2)
        next_t = time.monotonic()
        for i in range(steps):
            t = (i + 1) / steps
            ip = _lerp3(sp, target_pos, t)
            iq = _quat_slerp(sq, eq, t)
            if arm == "left":
                self._send_ee_cmd(ip, iq, r_sp, r_sq)
            else:
                self._send_ee_cmd(l_sp, l_sq, ip, iq)
            next_t = _precise_sleep(next_t, SERVO_PERIOD)
        return True

    def _try_sdk_move_waist(self, positions: List[float], velocities: List[float]) -> bool:
        if hasattr(self.robot, "move_waist_joint"):
            try:
                ret = self.robot.move_waist_joint(list(positions), list(velocities))
                return ret in (None, 0, getattr(getattr(agibot_gdk, "GDKRes", object), "kSuccess", 0))
            except Exception as e:
                print(f"  ⚠️  move_waist_joint 失败，回退到 JointControlReq: {e}")
        return False

    def _try_sdk_move_head(self, positions: List[float], velocities: List[float]) -> bool:
        if hasattr(self.robot, "move_head_joint"):
            try:
                ret = self.robot.move_head_joint(list(positions), list(velocities))
                return ret in (None, 0, getattr(getattr(agibot_gdk, "GDKRes", object), "kSuccess", 0))
            except Exception as e:
                print(f"  ⚠️  move_head_joint 失败，回退到 JointControlReq: {e}")
        return False

    def _try_sdk_move_arms(self, arm_positions_14: List[float], arm_velocities_14: List[float]) -> bool:
        if hasattr(self.robot, "move_arm_joint"):
            try:
                ret = self.robot.move_arm_joint(list(arm_positions_14), list(arm_velocities_14))
                return ret in (None, 0, getattr(getattr(agibot_gdk, "GDKRes", object), "kSuccess", 0))
            except Exception as e:
                print(f"  ⚠️  move_arm_joint 失败，回退到 JointControlReq: {e}")
        return False

    def _fallback_move_waist_jointreq(self, positions: List[float], velocities: List[float], life_time: float) -> None:
        # 先尝试打包一次
        try:
            self._send_joint_cmd(WAIST_JOINT_NAMES, positions, velocity=velocities[0], life_time=life_time)
            return
        except Exception:
            pass

        # 打包失败再逐关节顺序发送
        errs: List[str] = []
        for nm, pos, vel in zip(WAIST_JOINT_NAMES, positions, velocities):
            try:
                self._send_joint_cmd([nm], [pos], velocity=vel, life_time=life_time)
                time.sleep(0.05)
            except Exception as e:
                errs.append(f"{nm}: {e}")
        if errs:
            raise RuntimeError(" / ".join(errs))

    def _fallback_move_head_jointreq(self, positions: List[float], velocities: List[float], life_time: float) -> None:
        try:
            self._send_joint_cmd(HEAD_JOINT_NAMES, positions, velocity=velocities[0], life_time=life_time)
            return
        except Exception:
            pass

        for nm, pos, vel in zip(HEAD_JOINT_NAMES, positions, velocities):
            self._send_joint_cmd([nm], [pos], velocity=vel, life_time=life_time)
            time.sleep(0.02)

    def _fallback_move_arms_jointreq(
        self,
        left_positions: List[float],
        right_positions: List[float],
        left_velocity: float,
        right_velocity: float,
        life_time: float,
    ) -> None:
        # 先尝试左右臂各自一条多关节请求
        left_exc = None
        right_exc = None

        try:
            self._send_joint_cmd(LEFT_ARM_JOINT_NAMES, left_positions, velocity=left_velocity, life_time=life_time)
        except Exception as e:
            left_exc = e

        try:
            self._send_joint_cmd(RIGHT_ARM_JOINT_NAMES, right_positions, velocity=right_velocity, life_time=life_time)
        except Exception as e:
            right_exc = e

        if left_exc is None and right_exc is None:
            return

        # 再退到逐关节线程发送
        barrier = threading.Barrier(2)

        def _send_left():
            barrier.wait()
            for nm, pos in zip(LEFT_ARM_JOINT_NAMES, left_positions):
                self._send_joint_cmd([nm], [pos], velocity=left_velocity, life_time=life_time)
                time.sleep(0.01)

        def _send_right():
            barrier.wait()
            for nm, pos in zip(RIGHT_ARM_JOINT_NAMES, right_positions):
                self._send_joint_cmd([nm], [pos], velocity=right_velocity, life_time=life_time)
                time.sleep(0.01)

        t_l = threading.Thread(target=_send_left, daemon=True)
        t_r = threading.Thread(target=_send_right, daemon=True)
        t_l.start()
        t_r.start()
        t_l.join()
        t_r.join()

    def _move_arm_joints(
        self,
        arm: str,
        positions: List[float],
        velocity: float,
        wait: bool,
        wait_tol: float,
        wait_timeout: float,
    ) -> bool:
        names = LEFT_ARM_JOINT_NAMES if arm == "left" else RIGHT_ARM_JOINT_NAMES
        other_names = RIGHT_ARM_JOINT_NAMES if arm == "left" else LEFT_ARM_JOINT_NAMES
        label = "左臂" if arm == "left" else "右臂"

        if len(positions) != 7:
            print(f"❌ {label}需要 7 个值，传入 {len(positions)} 个")
            return False

        safe_pos = [_clamp_joint(nm, pos) for nm, pos in zip(names, positions)]
        target = {nm: sp for nm, sp in zip(names, safe_pos)}
        current = self._get_joint_map()
        other_pos = [current.get(nm, 0.0) for nm in other_names]

        if arm == "left":
            all_pos = safe_pos + other_pos
            all_vel = [velocity] * 7 + [0.02] * 7
            left_positions = safe_pos
            right_positions = other_pos
            left_velocity = velocity
            right_velocity = 0.02
        else:
            all_pos = other_pos + safe_pos
            all_vel = [0.02] * 7 + [velocity] * 7
            left_positions = other_pos
            right_positions = safe_pos
            left_velocity = 0.02
            right_velocity = velocity

        print(f"  → {label}关节运动: {[f'{p:+.3f}' for p in safe_pos]}")
        lt = float(wait_timeout)

        with self._lock:
            ok_sdk = self._try_sdk_move_arms(all_pos, all_vel)
            if not ok_sdk:
                self._fallback_move_arms_jointreq(
                    left_positions=left_positions,
                    right_positions=right_positions,
                    left_velocity=left_velocity,
                    right_velocity=right_velocity,
                    life_time=lt,
                )

        if wait:
            ok = self._wait_arrive(target, tol=wait_tol, timeout=wait_timeout)
            if ok:
                print(f"  ✅ {label}运动完成")
            return ok
        return True

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def get_all_joint_positions(self) -> Dict[str, float]:
        if not self._check_init():
            return {}
        return self._get_joint_map()

    def get_waist_positions(self) -> Optional[List[float]]:
        if not self._check_init():
            return None
        cur = self._get_joint_map()
        positions = []
        for name in WAIST_JOINT_NAMES:
            if name not in cur:
                print(f"❌ 无法读取 {name}")
                return None
            positions.append(cur[name])

        labels = ["joint1(俯仰)", "joint2(升降)", "joint3(俯仰)", "joint4(侧倾)", "joint5(旋转)"]
        print("  📐 当前腰部关节位置：")
        for lb, nm, p in zip(labels, WAIST_JOINT_NAMES, positions):
            lo, hi = JOINT_LIMITS.get(nm, (-999, 999))
            pct = (p - lo) / (hi - lo) * 100 if (hi - lo) > 1e-6 else 0.0
            print(f"    {lb:16s}  {p:+.4f} rad  ({math.degrees(p):+7.2f}°)  [{lo:.3f}~{hi:.3f}]  {pct:.0f}%")
        return positions

    def get_left_arm_positions(self) -> Optional[List[float]]:
        return self._get_arm_pos("left")

    def get_right_arm_positions(self) -> Optional[List[float]]:
        return self._get_arm_pos("right")

    def _get_arm_pos(self, arm: str) -> Optional[List[float]]:
        if not self._check_init():
            return None
        names = LEFT_ARM_JOINT_NAMES if arm == "left" else RIGHT_ARM_JOINT_NAMES
        label = "左臂" if arm == "left" else "右臂"
        cur = self._get_joint_map()
        positions = []
        for name in names:
            if name not in cur:
                print(f"❌ 无法读取 {name}")
                return None
            positions.append(cur[name])
        print(f"  📐 当前{label}关节位置：")
        for i, (nm, p) in enumerate(zip(names, positions)):
            print(f"    joint{i+1}  {p:+.4f} rad  ({math.degrees(p):+7.2f}°)")
        return positions

    def get_ee_pose(self, arm: str = "left") -> Tuple[List[float], List[float]]:
        if not self._check_init():
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
        if self.tf is None:
            print("❌ TF 不可用")
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
        return self._get_ee_tf(LEFT_EE_FRAME if arm == "left" else RIGHT_EE_FRAME)

    # ------------------------------------------------------------------
    # 腰部
    # ------------------------------------------------------------------

    def record_waist_point(self, name: str) -> bool:
        pos = self.get_waist_positions()
        if pos is None:
            return False
        self._ws.set(name, pos)
        return True

    def move_waist_to_point(
        self,
        name: str,
        velocities: Optional[List[float]] = None,
        wait: bool = True,
        wait_tol: float = 0.05,
        wait_timeout: float = 10.0,
    ) -> bool:
        if not self._check_init():
            return False
        pos = self._ws.get(name)
        if pos is None:
            print(f"❌ 腰部点位 '{name}' 不存在")
            self._ws.list_all()
            return False
        return self.move_waist(pos, velocities=velocities, wait=wait, wait_tol=wait_tol, wait_timeout=wait_timeout)

    def delete_waist_point(self, name: str) -> bool:
        return self._ws.delete(name)

    def list_waist_points(self) -> None:
        self._ws.list_all()

    def move_waist(
        self,
        positions: List[float],
        velocities: Optional[List[float]] = None,
        wait: bool = True,
        wait_tol: float = 0.05,
        wait_timeout: float = 10.0,
    ) -> bool:
        if not self._check_init():
            return False
        if len(positions) != 5:
            print(f"❌ 腰部需要 5 个值，传入了 {len(positions)} 个")
            return False

        vels = list(velocities) if velocities else [0.3] * 5
        if len(vels) == 1:
            vels = vels * 5
        if len(vels) != 5:
            vels = [0.3] * 5

        safe_pos = [_clamp_joint(nm, pos) for nm, pos in zip(WAIST_JOINT_NAMES, positions)]
        target_map = {nm: sp for nm, sp in zip(WAIST_JOINT_NAMES, safe_pos)}
        print(f"  → 腰部运动: {[f'{p:+.3f}' for p in safe_pos]}")

        with self._lock:
            ok_sdk = self._try_sdk_move_waist(safe_pos, vels)
            if not ok_sdk:
                try:
                    self._fallback_move_waist_jointreq(safe_pos, vels, life_time=float(wait_timeout))
                except Exception as e:
                    print(f"  ❌ 腰部运动失败: {e}")
                    return False

        if wait:
            ok = self._wait_arrive(target_map, tol=wait_tol, timeout=wait_timeout)
            if ok:
                print("  ✅ 腰部运动完成")
            return ok
        return True

    def move_waist_single(self, joint_index: int, position: float, velocity: float = 0.3, wait: bool = True) -> bool:
        if not self._check_init():
            return False
        if not 0 <= joint_index <= 4:
            print(f"❌ 腰部关节索引 0~4，传入 {joint_index}")
            return False

        cur = self.get_waist_positions()
        if cur is None:
            return False
        cur[joint_index] = position
        return self.move_waist(cur, velocities=[velocity] * 5, wait=wait)

    def run_waist_sequence(self, names: List[str], velocity: float = 0.2, dwell_s: float = 0.5) -> bool:
        return self._run_seq(names, self.move_waist_to_point, velocity, dwell_s, "腰部")

    # ------------------------------------------------------------------
    # 左臂
    # ------------------------------------------------------------------

    def record_left_arm_point(self, name: str) -> bool:
        pos = self._get_arm_pos("left")
        if pos is None:
            return False
        self._la.set(name, pos)
        return True

    def move_left_arm_to_point(
        self,
        name: str,
        velocity: float = 0.2,
        wait: bool = True,
        wait_tol: float = 0.05,
        wait_timeout: float = 15.0,
    ) -> bool:
        if not self._check_init():
            return False
        pos = self._la.get(name)
        if pos is None:
            print(f"❌ 左臂点位 '{name}' 不存在")
            self._la.list_all()
            return False
        return self._move_arm_joints("left", pos, velocity, wait, wait_tol, wait_timeout)

    def move_left_arm_servo(
        self,
        target_pos: List[float],
        target_quat: Optional[List[float]] = None,
        duration_s: float = 2.0,
    ) -> bool:
        if not self._check_init():
            return False
        sp, _ = self._get_ee_tf(LEFT_EE_FRAME)
        print(
            f"  → 左臂伺服: ({sp[0]:.3f},{sp[1]:.3f},{sp[2]:.3f})"
            f" → ({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f})"
            f"  {max(duration_s, 0.5):.1f}s"
        )
        with self._lock:
            ok = self._servo_move("left", target_pos, target_quat, max(duration_s, 0.5))
        if ok:
            print("  ✅ 左臂伺服完成")
        return ok

    def move_left_arm_delta(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0, duration_s: float = 2.0) -> bool:
        if not self._check_init():
            return False
        pos, quat = self.get_ee_pose("left")
        return self.move_left_arm_servo(
            target_pos=[pos[0] + dx, pos[1] + dy, pos[2] + dz],
            target_quat=quat,
            duration_s=duration_s,
        )

    def record_left_arm_ee_point(self, name: str) -> bool:
        if not self._check_init():
            return False
        if self.tf is None:
            print("❌ TF 不可用")
            return False
        pos, quat = self._get_ee_tf(LEFT_EE_FRAME)
        self._la.set("_ee_" + name, pos + quat)
        print(f"  📍 左臂末端点位 '{name}': ({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")
        return True

    def move_left_arm_ee_to_point(self, name: str, duration_s: float = 2.0) -> bool:
        if not self._check_init():
            return False
        data = self._la.get("_ee_" + name)
        if data is None:
            print(f"❌ 左臂末端点位 '{name}' 不存在，请先调用 record_left_arm_ee_point('{name}')")
            return False
        return self.move_left_arm_servo(data[:3], data[3:], duration_s=duration_s)

    def delete_left_arm_point(self, name: str) -> bool:
        if name in self._la:
            return self._la.delete(name)
        if ("_ee_" + name) in self._la:
            return self._la.delete("_ee_" + name)
        print(f"  ⚠️  左臂点位 '{name}' 不存在，可用: {self._la.keys()}")
        return False

    def list_left_arm_points(self) -> None:
        self._la.list_all()

    def run_left_arm_sequence(self, names: List[str], velocity: float = 0.2, dwell_s: float = 0.5) -> bool:
        return self._run_seq(names, self.move_left_arm_to_point, velocity, dwell_s, "左臂")

    # ------------------------------------------------------------------
    # 右臂
    # ------------------------------------------------------------------

    def record_right_arm_point(self, name: str) -> bool:
        pos = self._get_arm_pos("right")
        if pos is None:
            return False
        self._ra.set(name, pos)
        return True

    def move_right_arm_to_point(
        self,
        name: str,
        velocity: float = 0.2,
        wait: bool = True,
        wait_tol: float = 0.05,
        wait_timeout: float = 15.0,
    ) -> bool:
        if not self._check_init():
            return False
        pos = self._ra.get(name)
        if pos is None:
            print(f"❌ 右臂点位 '{name}' 不存在")
            self._ra.list_all()
            return False
        return self._move_arm_joints("right", pos, velocity, wait, wait_tol, wait_timeout)

    def move_right_arm_servo(
        self,
        target_pos: List[float],
        target_quat: Optional[List[float]] = None,
        duration_s: float = 2.0,
    ) -> bool:
        if not self._check_init():
            return False
        sp, _ = self._get_ee_tf(RIGHT_EE_FRAME)
        print(
            f"  → 右臂伺服: ({sp[0]:.3f},{sp[1]:.3f},{sp[2]:.3f})"
            f" → ({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f})"
            f"  {max(duration_s, 0.5):.1f}s"
        )
        with self._lock:
            ok = self._servo_move("right", target_pos, target_quat, max(duration_s, 0.5))
        if ok:
            print("  ✅ 右臂伺服完成")
        return ok

    def move_right_arm_delta(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0, duration_s: float = 2.0) -> bool:
        if not self._check_init():
            return False
        pos, quat = self.get_ee_pose("right")
        return self.move_right_arm_servo(
            target_pos=[pos[0] + dx, pos[1] + dy, pos[2] + dz],
            target_quat=quat,
            duration_s=duration_s,
        )

    def record_right_arm_ee_point(self, name: str) -> bool:
        if not self._check_init():
            return False
        if self.tf is None:
            print("❌ TF 不可用")
            return False
        pos, quat = self._get_ee_tf(RIGHT_EE_FRAME)
        self._ra.set("_ee_" + name, pos + quat)
        print(f"  📍 右臂末端点位 '{name}': ({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")
        return True

    def move_right_arm_ee_to_point(self, name: str, duration_s: float = 2.0) -> bool:
        if not self._check_init():
            return False
        data = self._ra.get("_ee_" + name)
        if data is None:
            print(f"❌ 右臂末端点位 '{name}' 不存在，请先调用 record_right_arm_ee_point('{name}')")
            return False
        return self.move_right_arm_servo(data[:3], data[3:], duration_s=duration_s)

    def delete_right_arm_point(self, name: str) -> bool:
        if name in self._ra:
            return self._ra.delete(name)
        if ("_ee_" + name) in self._ra:
            return self._ra.delete("_ee_" + name)
        print(f"  ⚠️  右臂点位 '{name}' 不存在，可用: {self._ra.keys()}")
        return False

    def list_right_arm_points(self) -> None:
        self._ra.list_all()

    def run_right_arm_sequence(self, names: List[str], velocity: float = 0.2, dwell_s: float = 0.5) -> bool:
        return self._run_seq(names, self.move_right_arm_to_point, velocity, dwell_s, "右臂")

    # ------------------------------------------------------------------
    # 双臂关节点位（同时记录 / 同时运动）
    # ------------------------------------------------------------------

    def record_both_arms_point(self, name: str) -> bool:
        if not self._check_init():
            return False

        l_pos = self._get_arm_pos("left")
        r_pos = self._get_arm_pos("right")
        if l_pos is None or r_pos is None:
            print("❌ 读取关节位置失败，双臂点位未记录")
            return False

        self._la.set(name, l_pos)
        self._ra.set(name, r_pos)
        print(f"  ✅ 双臂点位 '{name}' 已记录并保存到磁盘（左臂+右臂）")
        return True

    def move_both_arms_to_point(
        self,
        name: str,
        velocity: float = 0.2,
        wait: bool = True,
        wait_tol: float = 0.05,
        wait_timeout: float = 15.0,
        sync_arrival: bool = False,
    ) -> bool:
        if not self._check_init():
            return False

        l_pos = self._la.get(name)
        r_pos = self._ra.get(name)

        if l_pos is None and r_pos is None:
            print(f"❌ 双臂点位 '{name}' 不存在，请先调用 record_both_arms_point('{name}')")
            print("  左臂可用点位:", self._la.keys())
            print("  右臂可用点位:", self._ra.keys())
            return False
        if l_pos is None:
            print(f"⚠️  左臂点位 '{name}' 不存在，仅执行右臂")
        if r_pos is None:
            print(f"⚠️  右臂点位 '{name}' 不存在，仅执行左臂")

        cur = self._get_joint_map()
        if l_pos is None:
            l_pos = [cur.get(nm, 0.0) for nm in LEFT_ARM_JOINT_NAMES]
        if r_pos is None:
            r_pos = [cur.get(nm, 0.0) for nm in RIGHT_ARM_JOINT_NAMES]

        l_safe = [_clamp_joint(nm, pos) for nm, pos in zip(LEFT_ARM_JOINT_NAMES, l_pos)]
        r_safe = [_clamp_joint(nm, pos) for nm, pos in zip(RIGHT_ARM_JOINT_NAMES, r_pos)]

        target_map = {}
        for nm, sp in zip(LEFT_ARM_JOINT_NAMES, l_safe):
            target_map[nm] = sp
        for nm, sp in zip(RIGHT_ARM_JOINT_NAMES, r_safe):
            target_map[nm] = sp

        l_vel = velocity
        r_vel = velocity

        if sync_arrival:
            cur_map = self._get_joint_map()
            l_max_dist = max((abs(cur_map.get(nm, 0.0) - sp) for nm, sp in zip(LEFT_ARM_JOINT_NAMES, l_safe)), default=0.0)
            r_max_dist = max((abs(cur_map.get(nm, 0.0) - sp) for nm, sp in zip(RIGHT_ARM_JOINT_NAMES, r_safe)), default=0.0)
            max_dist = max(l_max_dist, r_max_dist)
            if max_dist > 1e-6:
                l_vel = max(0.02, velocity * (l_max_dist / max_dist)) if l_max_dist > 1e-6 else 0.02
                r_vel = max(0.02, velocity * (r_max_dist / max_dist)) if r_max_dist > 1e-6 else 0.02
            print(f"  [同步到达] 左臂速度={l_vel:.3f} rad/s  右臂速度={r_vel:.3f} rad/s")

        print(f"  → 双臂关节运动到点位: '{name}'  速度 {velocity} rad/s")

        all_pos = l_safe + r_safe
        all_vel = [l_vel] * 7 + [r_vel] * 7
        lt = float(wait_timeout)

        with self._lock:
            ok_sdk = self._try_sdk_move_arms(all_pos, all_vel)
            if not ok_sdk:
                try:
                    self._fallback_move_arms_jointreq(
                        left_positions=l_safe,
                        right_positions=r_safe,
                        left_velocity=l_vel,
                        right_velocity=r_vel,
                        life_time=lt,
                    )
                except Exception as e:
                    print(f"  ❌ 双臂运动失败: {e}")
                    return False

        if wait:
            ok = self._wait_arrive(target_map, tol=wait_tol, timeout=wait_timeout)
            if ok:
                print(f"  ✅ 双臂已到达点位 '{name}'")
            return ok
        print("  → 双臂指令已发送（非阻塞）")
        return True

    def delete_both_arms_point(self, name: str) -> bool:
        l_ok = self._la.delete(name) if name in self._la else False
        r_ok = self._ra.delete(name) if name in self._ra else False
        if not l_ok and not r_ok:
            print(f"  ⚠️  双臂均无点位 '{name}'")
            return False
        return True

    def list_both_arms_points(self) -> None:
        l_keys = set(k for k in self._la.keys() if not k.startswith("_ee_"))
        r_keys = set(k for k in self._ra.keys() if not k.startswith("_ee_"))
        paired = sorted(l_keys & r_keys)
        l_only = sorted(l_keys - r_keys)
        r_only = sorted(r_keys - l_keys)

        print("\n" + "─" * 50)
        print("  双臂关节点位总览")
        print("─" * 50)
        if paired:
            print("  ✅ 配对点位（左右臂均有，可用 move_both_arms_to_point）：")
            for n in paired:
                print(f"     {n}")
        if l_only:
            print("  🔵 仅左臂有：")
            for n in l_only:
                print(f"     {n}")
        if r_only:
            print("  🟠 仅右臂有：")
            for n in r_only:
                print(f"     {n}")
        if not paired and not l_only and not r_only:
            print("  （暂无关节点位）")
        print("─" * 50)

    def run_both_arms_sequence(self, names: List[str], velocity: float = 0.2, dwell_s: float = 0.5) -> bool:
        print(f"  → 双臂序列运动: {names}")
        for i, nm in enumerate(names):
            print(f"\n  [{i+1}/{len(names)}] → '{nm}'")
            if not self.move_both_arms_to_point(nm, velocity=velocity, wait=True):
                print(f"  ❌ 到达 '{nm}' 失败，序列中止")
                return False
            if dwell_s > 0:
                time.sleep(dwell_s)
        print("  ✅ 双臂序列完成")
        return True

    # ------------------------------------------------------------------
    # 双臂末端点位（同时记录 / 同时伺服运动）
    # ------------------------------------------------------------------

    def record_both_arms_ee_point(self, name: str) -> bool:
        if not self._check_init():
            return False
        if self.tf is None:
            print("❌ TF 不可用，无法记录末端点位")
            return False

        l_pos, l_quat = self._get_ee_tf(LEFT_EE_FRAME)
        r_pos, r_quat = self._get_ee_tf(RIGHT_EE_FRAME)

        self._la.set("_ee_" + name, l_pos + l_quat)
        self._ra.set("_ee_" + name, r_pos + r_quat)

        print(f"  ✅ 双臂末端点位 '{name}' 已记录并保存到磁盘")
        print(f"     左臂: ({l_pos[0]:.3f}, {l_pos[1]:.3f}, {l_pos[2]:.3f})")
        print(f"     右臂: ({r_pos[0]:.3f}, {r_pos[1]:.3f}, {r_pos[2]:.3f})")
        return True

    def move_both_arms_ee_to_point(self, name: str, duration_s: float = 2.0) -> bool:
        if not self._check_init():
            return False
        if self.tf is None:
            print("❌ TF 不可用")
            return False

        l_data = self._la.get("_ee_" + name)
        r_data = self._ra.get("_ee_" + name)

        if l_data is None and r_data is None:
            print(f"❌ 双臂末端点位 '{name}' 不存在，请先调用 record_both_arms_ee_point('{name}')")
            return False
        if l_data is None:
            print(f"⚠️  左臂末端点位 '{name}' 不存在，仅执行右臂")
        if r_data is None:
            print(f"⚠️  右臂末端点位 '{name}' 不存在，仅执行左臂")

        l_tp = l_data[:3] if l_data else None
        l_tq = l_data[3:] if l_data else None
        r_tp = r_data[:3] if r_data else None
        r_tq = r_data[3:] if r_data else None

        print(f"  → 双臂末端伺服到点位: '{name}'  {max(duration_s, 0.5):.1f}s")
        return self.move_both_arms_servo(
            left_pos=l_tp,
            left_quat=l_tq,
            right_pos=r_tp,
            right_quat=r_tq,
            duration_s=max(duration_s, 0.5),
        )
    

    def move_both_arms_ee_to_point_with_offset(
        self,
        name: str,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        duration_s: float = 2.0,
    ) -> bool:
        """
        在已记录的双臂末端点位基础上，加同一个 xyz 偏移后执行同步伺服。
        适合做视觉平移修正。
        """
        if not self._check_init():
            return False

        data = self.get_both_arms_ee_point(name)
        if data is None:
            return False

        left_pos = [
            data["left_pos"][0] + dx,
            data["left_pos"][1] + dy,
            data["left_pos"][2] + dz,
        ]
        right_pos = [
            data["right_pos"][0] + dx,
            data["right_pos"][1] + dy,
            data["right_pos"][2] + dz,
        ]

        print(
            f"  → 双臂末端点位 '{name}' 加偏移运动: "
            f"dx={dx:+.3f}, dy={dy:+.3f}, dz={dz:+.3f}"
        )

        return self.move_both_arms_servo(
            left_pos=left_pos,
            left_quat=data["left_quat"],
            right_pos=right_pos,
            right_quat=data["right_quat"],
            duration_s=duration_s,
        )





    def get_both_arms_ee_point(self, name: str):
        """
        读取双臂末端点位数据。

        返回格式:
        {
            "left_pos": [x, y, z],
            "left_quat": [qx, qy, qz, qw],
            "right_pos": [x, y, z],
            "right_quat": [qx, qy, qz, qw],
        }
        """
        key = "_ee_" + name

        l_data = self._la.get(key)
        r_data = self._ra.get(key)

        if l_data is None and r_data is None:
            print(f"❌ 双臂末端点位 '{name}' 不存在")
            return None
        if l_data is None:
            print(f"❌ 左臂末端点位 '{name}' 不存在")
            return None
        if r_data is None:
            print(f"❌ 右臂末端点位 '{name}' 不存在")
            return None

        if len(l_data) != 7 or len(r_data) != 7:
            print(f"❌ 双臂末端点位 '{name}' 数据格式错误")
            return None

        return {
            "left_pos": [float(v) for v in l_data[:3]],
            "left_quat": [float(v) for v in l_data[3:]],
            "right_pos": [float(v) for v in r_data[:3]],
            "right_quat": [float(v) for v in r_data[3:]],
        }

        #加一个“带偏移的双臂末端点位运动”方法
        #ctrl.move_both_arms_ee_to_point_with_offset("A1", dx=..., dy=..., dz=..., duration_s=2.0)







    def delete_both_arms_ee_point(self, name: str) -> bool:
        key = "_ee_" + name
        l_ok = self._la.delete(key) if key in self._la else False
        r_ok = self._ra.delete(key) if key in self._ra else False
        if not l_ok and not r_ok:
            print(f"  ⚠️  双臂末端点位 '{name}' 不存在")
            return False
        return True

    def list_both_arms_ee_points(self) -> None:
        l_ee = set(k[4:] for k in self._la.keys() if k.startswith("_ee_"))
        r_ee = set(k[4:] for k in self._ra.keys() if k.startswith("_ee_"))
        paired = sorted(l_ee & r_ee)
        l_only = sorted(l_ee - r_ee)
        r_only = sorted(r_ee - l_ee)

        print("\n" + "─" * 50)
        print("  双臂末端点位总览")
        print("─" * 50)
        if paired:
            print("  ✅ 配对点位（可用 move_both_arms_ee_to_point）：")
            for n in paired:
                ld = self._la.get("_ee_" + n)
                rd = self._ra.get("_ee_" + n)
                lp = f"({ld[0]:.3f},{ld[1]:.3f},{ld[2]:.3f})" if ld else "?"
                rp = f"({rd[0]:.3f},{rd[1]:.3f},{rd[2]:.3f})" if rd else "?"
                print(f"     {n:20s}  左:{lp}  右:{rp}")
        if l_only:
            print(f"  🔵 仅左臂有：{l_only}")
        if r_only:
            print(f"  🟠 仅右臂有：{r_only}")
        if not paired and not l_only and not r_only:
            print("  （暂无末端点位）")
        print("─" * 50)

    # ------------------------------------------------------------------
    # 双臂同步伺服
    # ------------------------------------------------------------------

    def move_both_arms_servo(
        self,
        left_pos: Optional[List[float]] = None,
        left_quat: Optional[List[float]] = None,
        right_pos: Optional[List[float]] = None,
        right_quat: Optional[List[float]] = None,
        duration_s: float = 2.0,
    ) -> bool:
        if not self._check_init():
            return False
        if self.tf is None:
            print("❌ TF 不可用")
            return False
        if not self._ensure_servo_mode():
            return False

        duration_s = max(duration_s, 0.5)
        l_sp, l_sq = self._get_ee_tf(LEFT_EE_FRAME)
        r_sp, r_sq = self._get_ee_tf(RIGHT_EE_FRAME)

        l_ep = left_pos if left_pos is not None else l_sp
        r_ep = right_pos if right_pos is not None else r_sp
        l_eq = _quat_norm(left_quat) if left_quat else l_sq
        r_eq = _quat_norm(right_quat) if right_quat else r_sq

        steps = max(int(duration_s * SERVO_HZ), 2)
        next_t = time.monotonic()

        print(f"  → 双臂同步伺服 {duration_s:.1f}s")
        with self._lock:
            for i in range(steps):
                t = (i + 1) / steps
                lp = _lerp3(l_sp, l_ep, t)
                lq = _quat_slerp(l_sq, l_eq, t)
                rp = _lerp3(r_sp, r_ep, t)
                rq = _quat_slerp(r_sq, r_eq, t)
                self._send_ee_cmd(lp, lq, rp, rq)
                next_t = _precise_sleep(next_t, SERVO_PERIOD)

        print("  ✅ 双臂伺服完成")
        return True

    def move_both_arms_delta(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0, duration_s: float = 2.0) -> bool:
        if not self._check_init():
            return False
        if self.tf is None:
            print("❌ TF 不可用")
            return False

        l_pos, l_quat = self._get_ee_tf(LEFT_EE_FRAME)
        r_pos, r_quat = self._get_ee_tf(RIGHT_EE_FRAME)
        print(f"  → 双臂相对位移: dx={dx:+.3f} dy={dy:+.3f} dz={dz:+.3f} (m)  {duration_s:.1f}s")

        return self.move_both_arms_servo(
            left_pos=[l_pos[0] + dx, l_pos[1] + dy, l_pos[2] + dz],
            left_quat=l_quat,
            right_pos=[r_pos[0] + dx, r_pos[1] + dy, r_pos[2] + dz],
            right_quat=r_quat,
            duration_s=duration_s,
        )

    # ------------------------------------------------------------------
    # 头部
    # ------------------------------------------------------------------

    def move_head(
        self,
        positions: List[float],
        velocities: Optional[List[float]] = None,
        wait: bool = True,
        wait_tol: float = 0.03,
        wait_timeout: float = 5.0,
    ) -> bool:
        if not self._check_init():
            return False
        if len(positions) != 3:
            print("❌ 头部需要 3 个值")
            return False

        vels = list(velocities) if velocities else [0.3, 0.3, 0.3]
        if len(vels) == 1:
            vels = vels * 3
        if len(vels) != 3:
            vels = [0.3, 0.3, 0.3]

        safe = [_clamp_joint(nm, pos) for nm, pos in zip(HEAD_JOINT_NAMES, positions)]
        print(f"  → 头部: roll={safe[0]:.3f} pitch={safe[1]:.3f} yaw={safe[2]:.3f}")

        with self._lock:
            ok_sdk = self._try_sdk_move_head(safe, vels)
            if not ok_sdk:
                try:
                    self._fallback_move_head_jointreq(safe, vels, life_time=float(wait_timeout))
                except Exception as e:
                    print(f"  ❌ 头部运动失败: {e}")
                    return False

        if wait:
            target = {HEAD_JOINT_NAMES[i]: safe[i] for i in range(3)}
            ok = self._wait_arrive(target, tol=wait_tol, timeout=wait_timeout)
            if ok:
                print("  ✅ 头部运动完成")
            return ok
        return True

    def look_at_direction(self, pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0,
                        velocity: float = 0.3, wait: bool = True) -> bool:
        # 实机标定结果：
        # joint1 -> yaw
        # joint2 -> roll
        # joint3 -> pitch
        return self.move_head([yaw, roll, pitch], [velocity] * 3, wait=wait)

    def look_forward(self, velocity: float = 0.3, wait: bool = True) -> bool:
        return self.move_head([0.0, 0.0, 0.0], [velocity] * 3, wait=wait)

    # ------------------------------------------------------------------
    # 全身点位
    # ------------------------------------------------------------------

    def record_body_point(self, name: str) -> bool:
        if not self._check_init():
            return False
        cur = self._get_joint_map()
        vals = [cur.get(n, 0.0) for n in JOINT_NAMES]
        self._bd.set(name, vals)
        return True

    def move_to_body_point(
        self,
        name: str,
        velocity: float = 0.2,
        wait: bool = True,
        wait_tol: float = 0.05,
        wait_timeout: float = 20.0,
    ) -> bool:
        if not self._check_init():
            return False
        vals = self._bd.get(name)
        if vals is None:
            print(f"❌ 全身点位 '{name}' 不存在")
            self._bd.list_all()
            return False

        target = dict(zip(JOINT_NAMES, vals))
        print(f"  → 全身运动到 '{name}'")

        # 腰部 / 头部 / 双臂拆开执行，兼容性更好
        waist = vals[0:5]
        head = vals[5:8]
        left = vals[8:15]
        right = vals[15:22]

        if not self.move_waist(waist, velocities=[velocity] * 5, wait=False, wait_timeout=wait_timeout):
            return False
        if not self.move_head(head, velocities=[velocity] * 3, wait=False, wait_timeout=wait_timeout):
            return False

        current = self._get_joint_map()
        for nm, p in zip(LEFT_ARM_JOINT_NAMES, left):
            target[nm] = p
        for nm, p in zip(RIGHT_ARM_JOINT_NAMES, right):
            target[nm] = p

        with self._lock:
            ok_sdk = self._try_sdk_move_arms(left + right, [velocity] * 14)
            if not ok_sdk:
                try:
                    self._fallback_move_arms_jointreq(left, right, velocity, velocity, float(wait_timeout))
                except Exception as e:
                    print(f"  ❌ 全身双臂部分失败: {e}")
                    return False

        if wait:
            ok = self._wait_arrive(target, tol=wait_tol, timeout=wait_timeout)
            if ok:
                print(f"  ✅ 全身到达 '{name}'")
            return ok
        return True

    # 兼容别名
    move_body_to_point = move_to_body_point

    def delete_body_point(self, name: str) -> bool:
        return self._bd.delete(name)

    def list_body_points(self) -> None:
        ks = self._bd.keys()
        if not ks:
            print("  （暂无全身点位）")
        else:
            print(f"\n  [全身] 共 {len(ks)} 个点位: {ks}")

    # ------------------------------------------------------------------
    # 夹爪
    # ------------------------------------------------------------------

    def open_gripper(self, arm: str = "both", position: float = 0.0) -> bool:
        return self._gripper(arm, position)

    def close_gripper(self, arm: str = "both", position: float = 1.0) -> bool:
        return self._gripper(arm, position)

    def _gripper(self, arm: str, position: float) -> bool:
        if not self._check_init():
            return False

        pos = float(_clamp(position, 0.0, 1.0))

        try:
            js = agibot_gdk.JointStates()
            js.target_type = "omnipicker"

            if arm == "both":
                js.group = "dual_tool"
                s_l = agibot_gdk.JointState()
                s_l.position = pos
                s_r = agibot_gdk.JointState()
                s_r.position = pos
                js.states = [s_l, s_r]
            elif arm == "left":
                js.group = "left_tool"
                s = agibot_gdk.JointState()
                s.position = pos
                js.states = [s]
            elif arm == "right":
                js.group = "right_tool"
                s = agibot_gdk.JointState()
                s.position = pos
                js.states = [s]
            else:
                print("❌ arm 只能是 'left' / 'right' / 'both'")
                return False

            js.nums = len(js.states)
            self.robot.move_ee_pos(js)
            print(f"  ✅ 夹爪[{arm}] → {pos:.2f}  (omnipicker: 0=开 1=关)")
            return True
        except Exception as e:
            print(f"  ❌ 夹爪失败: {e}")
            return False

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def list_all_points(self) -> None:
        print("\n" + "=" * 60)
        print("  所有点位总览")
        print("=" * 60)
        self._ws.list_all()
        self._la.list_all()
        self._ra.list_all()
        self.list_body_points()
        print("=" * 60)

    def home_all(self, velocity: float = 0.05) -> bool:
        if not self._check_init():
            return False
        print(f"  ⚠️  全身回零（速度 {velocity} rad/s），请确认周围安全！")
        return self.move_to_body_point("__ZERO__", velocity=velocity) if "__ZERO__" in self._bd else False

    def _run_seq(self, names: List[str], move_fn, velocity: float, dwell_s: float, label: str) -> bool:
        print(f"  → [{label}] 序列: {names}")
        for i, nm in enumerate(names):
            print(f"  [{i+1}/{len(names)}] → '{nm}'")
            if not move_fn(nm, velocity=velocity, wait=True):
                print(f"  ❌ '{nm}' 失败，序列中止")
                return False
            if dwell_s > 0:
                time.sleep(dwell_s)
        print(f"  ✅ [{label}] 序列完成")
        return True


if __name__ == "__main__":
    import sys

    ctrl = G2RobotController()
    if not ctrl.init():
        sys.exit(1)

    try:
        ctrl.look_at_direction(pitch=0.15)
        time.sleep(0.3)
        ctrl.look_forward()

        ctrl.record_waist_point("waist_home")
        ctrl.record_left_arm_point("left_ready")
        ctrl.record_right_arm_point("right_ready")
        ctrl.record_both_arms_point("both_ready")

        ctrl.record_left_arm_ee_point("left_pick")
        ctrl.record_right_arm_ee_point("right_pick")
        ctrl.record_both_arms_ee_point("both_pick")

        ctrl.list_both_arms_points()
        ctrl.list_both_arms_ee_points()
        ctrl.list_all_points()

    except KeyboardInterrupt:
        print("\n⚠️  中断")
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        ctrl.release()