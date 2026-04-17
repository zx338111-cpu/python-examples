#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robot_controller.py — G2 机器人底盘/导航控制封装

修复重点：
1. 申请底盘控制前，先检查/取消当前 PNC 活动任务
2. 停车后，等待停稳并清理任务状态
3. 导航与底盘控制统一走稳定的任务状态收尾逻辑
"""

import json
import math
import os
import signal
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import agibot_gdk

_S: Dict[int, str] = {
    0: "空闲",
    1: "启动中",
    2: "运行中",
    3: "暂停中",
    4: "已暂停",
    5: "恢复中",
    6: "取消中",
    7: "已取消",
    8: "失败",
    9: "成功",
}
_IDLE_OR_DONE = {0, 7, 8, 9}

DEFAULT_LINEAR_SPEED = 0.25
DEFAULT_ANGULAR_SPEED = 0.4
CTRL_HZ = 20
CTRL_DT = 1.0 / CTRL_HZ
STOP_SPEED_THRESH = 0.05


class RobotController:
    def __init__(self,
                 waypoints_json: Optional[str] = None,
                 verbose: bool = True):
        self._verbose = verbose
        self._chassis_mode = -1

        self._log("正在初始化 GDK...")
        init_res = agibot_gdk.gdk_init()
        if init_res != agibot_gdk.GDKRes.kSuccess:
            raise RuntimeError(f"GDK 初始化失败: {init_res}")

        self.pnc = agibot_gdk.Pnc()
        self.slam = agibot_gdk.Slam()
        time.sleep(1.5)

        self._log("正在从地图读取导航点...")
        self.waypoints, self.map_id = self._load_map_waypoints()
        self._log(f"地图 id={self.map_id}，读到 {len(self.waypoints)} 个导航点")

        json_path = waypoints_json or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "waypoints.json"
        )
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                extra = json.load(f)
            for k, v in extra.items():
                if k not in self.waypoints:
                    v["source"] = "json"
                    self.waypoints[k] = v
            self._log(f"已合并 {json_path}，共 {len(self.waypoints)} 个导航点")

        self._wp_list = list(self.waypoints.keys())

        signal.signal(signal.SIGINT, self._on_exit)
        signal.signal(signal.SIGTERM, self._on_exit)

    def release(self):
        try:
            self._stop_chassis()
        except Exception:
            pass
        try:
            self._cancel_active_task(wait_timeout=1.5)
        except Exception:
            pass
        try:
            agibot_gdk.gdk_release()
            self._log("✅ GDK 已释放")
        except Exception as e:
            self._log(f"⚠️ GDK 释放失败: {e}")

    def _log(self, msg: str):
        if self._verbose:
            print(msg)

    def _on_exit(self, *_):
        print("\n收到退出信号，停止机器人...")
        try:
            self.release()
        finally:
            sys.exit(0)

    def _load_map_waypoints(self) -> Tuple[Dict[str, dict], int]:
        m = agibot_gdk.Map()
        time.sleep(1.0)
        curr = m.get_curr_map()
        result = m.get_map(curr.id)

        waypoints: Dict[str, dict] = {}
        for pt in result.guide_pts:
            name = str(pt.id)
            waypoints[name] = {
                "position": [pt.pt.position.x, pt.pt.position.y, pt.pt.position.z],
                "orientation": [
                    pt.pt.orientation.x,
                    pt.pt.orientation.y,
                    pt.pt.orientation.z,
                    pt.pt.orientation.w,
                ],
                "type": pt.type,
                "source": "map",
            }
        return waypoints, curr.id

    def _task_state(self) -> Tuple[int, int, str]:
        ts = self.pnc.get_task_state()
        return ts.state, ts.id, getattr(ts, "message", "")

    def _wait_task_idle(self, timeout: float = 3.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            state, task_id, msg = self._task_state()
            if state in _IDLE_OR_DONE:
                return True
            time.sleep(0.1)
        return False

    def _cancel_active_task(self, wait_timeout: float = 3.0):
        try:
            state, task_id, msg = self._task_state()
            if state not in _IDLE_OR_DONE:
                self._log(
                    f"   发现活动任务: state={state}({_S.get(state, state)}), "
                    f"id={task_id}, msg={msg}，先取消"
                )
                try:
                    self.pnc.cancel_task(task_id)
                except Exception as e:
                    self._log(f"   cancel_task 失败: {e}")
                self._wait_task_idle(wait_timeout)
        except Exception as e:
            self._log(f"   查询/取消任务失败: {e}")

    def _cancel_navi(self):
        self._cancel_active_task(wait_timeout=3.0)

    def _request_chassis(self, mode: int):
        self._cancel_active_task(wait_timeout=3.0)

        state, task_id, msg = self._task_state()
        if state not in _IDLE_OR_DONE:
            raise RuntimeError(
                f"底盘当前任务未结束，无法申请控制权: "
                f"state={state}({_S.get(state, state)}), id={task_id}, msg={msg}"
            )

        self.pnc.request_chassis_control(mode)
        time.sleep(0.3)
        self._chassis_mode = mode

    def _make_twist(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0) -> agibot_gdk.Twist:
        twist = agibot_gdk.Twist()
        twist.linear = agibot_gdk.Vector3()
        twist.angular = agibot_gdk.Vector3()
        twist.linear.x = vx
        twist.linear.y = vy
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = wz
        return twist

    def _stop_chassis(self):
        try:
            twist = self._make_twist()
            for _ in range(5):
                self.pnc.move_chassis(twist)
                time.sleep(0.05)
        except Exception:
            pass

        self._wait_stop()

        try:
            self._cancel_active_task(wait_timeout=2.0)
        except Exception:
            pass

        self._chassis_mode = -1

    def _get_speed(self) -> float:
        try:
            odom = self.slam.get_odom_info()
            return math.hypot(odom.velocity.x, odom.velocity.y)
        except Exception:
            return 0.0

    def _wait_stop(self, timeout: float = 3.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._get_speed() < STOP_SPEED_THRESH:
                break
            time.sleep(0.1)

    def _resolve_wp(self, key: Union[int, str]) -> Optional[str]:
        if isinstance(key, int):
            if 0 <= key < len(self._wp_list):
                return self._wp_list[key]
            return None

        try:
            idx = int(key)
            if 0 <= idx < len(self._wp_list):
                return self._wp_list[idx]
        except ValueError:
            pass

        return key if key in self.waypoints else None

    def list_waypoints(self):
        print(f"\n{'─'*62}")
        print(f"{'索引':<6} {'名称':<10} {'来源':<6} {'位置 (x, y)'}")
        print(f"{'─'*62}")
        for i, name in enumerate(self._wp_list):
            wp = self.waypoints[name]
            pos = wp["position"]
            src = wp.get("source", "?")
            print(f"[{i}]    {name:<10} {src:<6} ({pos[0]:.4f}, {pos[1]:.4f})")
        print(f"{'─'*62}")
        print(f"共 {len(self.waypoints)} 个导航点\n")

    def get_current_pose(self) -> dict:
        odom = self.slam.get_odom_info()
        pos = odom.pose.pose.position
        ori = odom.pose.pose.orientation
        return {
            "position": [pos.x, pos.y, pos.z],
            "orientation": [ori.x, ori.y, ori.z, ori.w],
            "loc_state": odom.loc_state,
            "loc_confidence": odom.loc_confidence,
        }

    def go(self,
           waypoint: Union[int, str],
           high_precision: bool = False,
           timeout: float = 120.0) -> bool:
        name = self._resolve_wp(waypoint)
        if name is None:
            self._log(f"❌ 找不到导航点: '{waypoint}'")
            return False

        wp = self.waypoints[name]
        pos = wp["position"]
        ori = wp["orientation"]
        src = wp.get("source", "?")
        mode_str = "高精度" if high_precision else "普通"

        state, _, _ = self._task_state()
        if state not in _IDLE_OR_DONE:
            self._log("   取消旧任务...")
            self._cancel_navi()

        req = agibot_gdk.NaviReq()
        req.target.position.x = pos[0]
        req.target.position.y = pos[1]
        req.target.position.z = pos[2]
        req.target.orientation.x = ori[0]
        req.target.orientation.y = ori[1]
        req.target.orientation.z = ori[2]
        req.target.orientation.w = ori[3]

        self._log(f"\n🚀 导航到: '{name}' [{src}] ({mode_str})")
        self._log(f"   目标: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

        try:
            if high_precision:
                self.pnc.high_precision_navi(req)
            else:
                self.pnc.normal_navi(req)
        except Exception as e:
            self._log(f"❌ 发送失败: {e}")
            return False

        self._log("   等待启动...")
        started = False
        for _ in range(20):
            time.sleep(0.5)
            state, _, msg = self._task_state()
            if state == 2:
                started = True
                self._log("   已启动")
                break
            if state in _IDLE_OR_DONE:
                break

        if not started:
            state, _, msg = self._task_state()
            if state == 9:
                self._log(f"✅ '{name}' 已在目标点附近，无需移动")
                return True
            self._log(f"❌ 任务未能启动: {_S.get(state, state)}  {msg}")
            return False

        start = time.time()
        while time.time() - start < timeout:
            time.sleep(0.5)
            state, _, msg = self._task_state()
            elapsed = time.time() - start
            if self._verbose:
                print(f"\r   {_S.get(state, state)}... {elapsed:.0f}s/{timeout:.0f}s",
                      end="", flush=True)

            if state == 9:
                self._wait_stop()
                elapsed = time.time() - start
                self._log(f"\n✅ 已到达 '{name}'！耗时 {elapsed:.1f}s")
                return True

            if state in {7, 8}:
                self._log(f"\n❌ 导航失败: {_S.get(state, state)}  {msg}")
                return False

        self._log("\n⏰ 超时，取消任务")
        self._cancel_navi()
        return False

    def go_sequence(self,
                    waypoints: List[Union[int, str]],
                    high_precision: bool = False,
                    timeout: float = 120.0,
                    stop_on_fail: bool = True) -> bool:
        names: List[str] = []
        for wp in waypoints:
            name = self._resolve_wp(wp)
            if name is None:
                self._log(f"❌ 找不到导航点: '{wp}'")
                return False
            names.append(name)

        self._log(f"\n📍 序列导航: {' → '.join(names)}")
        all_ok = True
        for i, name in enumerate(names):
            self._log(f"\n[{i+1}/{len(names)}]")
            ok = self.go(name, high_precision, timeout)
            if not ok:
                all_ok = False
                if stop_on_fail:
                    self._log(f"⛔ 在 '{name}' 失败，停止序列")
                    break
                self._log(f"⚠️ 在 '{name}' 失败，继续下一个")
            if i < len(names) - 1:
                time.sleep(1.0)

        self._log("\n🎉 序列导航完成！" if all_ok else "\n⚠️ 序列导航未完全成功")
        return all_ok

    def crab_walk(self,
                  dist_m: float,
                  speed: float = DEFAULT_LINEAR_SPEED) -> bool:
        if abs(dist_m) < 0.01:
            return True

        self._request_chassis(1)
        direction = 1.0 if dist_m >= 0 else -1.0
        vy = direction * abs(speed)
        duration = abs(dist_m) / abs(speed)

        self._log(f"\n↔️  蟹行: {dist_m:+.2f}m  速度: {vy:+.2f}m/s  预计: {duration:.1f}s")

        twist = self._make_twist(vy=vy)
        start = time.time()

        try:
            while time.time() - start < duration:
                self.pnc.move_chassis(twist)
                time.sleep(CTRL_DT)
                if self._verbose:
                    elapsed = time.time() - start
                    print(f"\r   已移动: {elapsed/duration*abs(dist_m):.2f}m / {abs(dist_m):.2f}m",
                          end="", flush=True)
        except KeyboardInterrupt:
            self._log("\n⚠️  用户中断")
        finally:
            self._stop_chassis()

        self._log(f"\n✅ 蟹行完成，用时 {time.time()-start:.1f}s")
        return True

    def move_forward(self,
                     dist_m: float,
                     speed: float = DEFAULT_LINEAR_SPEED) -> bool:
        if abs(dist_m) < 0.01:
            return True

        self._request_chassis(0)
        direction = 1.0 if dist_m >= 0 else -1.0
        vx = direction * abs(speed)
        duration = abs(dist_m) / abs(speed)

        self._log(f"\n⬆️  前进: {dist_m:+.2f}m  速度: {vx:+.2f}m/s  预计: {duration:.1f}s")

        twist = self._make_twist(vx=vx)
        start = time.time()

        try:
            while time.time() - start < duration:
                self.pnc.move_chassis(twist)
                time.sleep(CTRL_DT)
                if self._verbose:
                    elapsed = time.time() - start
                    print(f"\r   已移动: {elapsed/duration*abs(dist_m):.2f}m / {abs(dist_m):.2f}m",
                          end="", flush=True)
        except KeyboardInterrupt:
            self._log("\n⚠️  用户中断")
        finally:
            self._stop_chassis()

        self._log(f"\n✅ 前进完成，用时 {time.time()-start:.1f}s")
        return True

    def rotate(self,
               angle_rad: float,
               speed: float = DEFAULT_ANGULAR_SPEED) -> bool:
        if abs(angle_rad) < 0.01:
            return True

        self._request_chassis(0)
        direction = 1.0 if angle_rad >= 0 else -1.0
        wz = direction * abs(speed)
        duration = abs(angle_rad) / abs(speed)

        deg = math.degrees(angle_rad)
        self._log(f"\n🔄  旋转: {deg:+.1f}°  角速度: {wz:+.2f}rad/s  预计: {duration:.1f}s")

        twist = self._make_twist(wz=wz)
        start = time.time()

        try:
            while time.time() - start < duration:
                self.pnc.move_chassis(twist)
                time.sleep(CTRL_DT)
                if self._verbose:
                    elapsed = time.time() - start
                    print(f"\r   已旋转: {elapsed/duration*abs(deg):.1f}° / {abs(deg):.1f}°",
                          end="", flush=True)
        except KeyboardInterrupt:
            self._log("\n⚠️  用户中断")
        finally:
            self._stop_chassis()

        self._log(f"\n✅ 旋转完成，用时 {time.time()-start:.1f}s")
        return True


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="G2 机器人底盘/导航控制")
    ap.add_argument("--list", action="store_true", help="列出所有导航点")
    ap.add_argument("--wp", type=str, help="导航到指定点（索引或名称）")
    ap.add_argument("--seq", type=str, nargs="+", help="序列导航")
    ap.add_argument("--high-precision", action="store_true", help="高精度导航")
    ap.add_argument("--crab", type=float, help="蟹行横移距离（米，正=左，负=右）")
    ap.add_argument("--forward", type=float, help="前进距离（米，正=前，负=后）")
    ap.add_argument("--rotate", type=float, help="旋转角度（度，正=左，负=右）")
    ap.add_argument("--speed", type=float, help="运动速度覆盖")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    robot = RobotController()

    try:
        if args.list:
            robot.list_waypoints()
        elif args.wp:
            robot.go(args.wp, args.high_precision, args.timeout)
        elif args.seq:
            robot.go_sequence(args.seq, args.high_precision, args.timeout)
        elif args.crab is not None:
            spd = args.speed or DEFAULT_LINEAR_SPEED
            robot.crab_walk(args.crab, spd)
        elif args.forward is not None:
            spd = args.speed or DEFAULT_LINEAR_SPEED
            robot.move_forward(args.forward, spd)
        elif args.rotate is not None:
            spd = args.speed or DEFAULT_ANGULAR_SPEED
            robot.rotate(math.radians(args.rotate), spd)
        else:
            robot.list_waypoints()
            print("用法: python3 robot_controller.py --wp 0")
            print("      python3 robot_controller.py --crab 1.5")
            print("      python3 robot_controller.py --forward 2.0")
            print("      python3 robot_controller.py --rotate 90")
    finally:
        robot.release()