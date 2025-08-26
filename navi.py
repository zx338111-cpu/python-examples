#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
navi.py — G2机器人导航脚本（自动读取地图导航点版）

自动从地图中读取 Pad 上设置的导航点，无需手动填写坐标。

用法：
  python3 navi.py --list          # 列出地图中所有导航点
  python3 navi.py --wp 0          # 导航到第0个导航点（按索引）
  python3 navi.py --wp 1          # 导航到第1个导航点
  python3 navi.py --seq 0 1       # 依次导航到多个点
"""

import argparse
import json
import os
import signal
import sys
import time

import agibot_gdk

# 状态码
S = {0:"空闲", 1:"启动中", 2:"运行中", 3:"暂停中",
     4:"已暂停", 5:"恢复中", 6:"取消中", 7:"已取消",
     8:"失败", 9:"成功"}
DONE_STATES = {7, 8, 9}


def load_map_waypoints():
    """从地图中读取 Pad 上设置的导航点"""
    m = agibot_gdk.Map()
    time.sleep(1.5)
    curr = m.get_curr_map()
    result = m.get_map(curr.id)
    waypoints = {}
    for pt in result.guide_pts:
        name = str(pt.id)
        waypoints[name] = {
            "position": [
                pt.pt.position.x,
                pt.pt.position.y,
                pt.pt.position.z,
            ],
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


class Navigator:
    def __init__(self):
        agibot_gdk.gdk_init()
        self.pnc  = agibot_gdk.Pnc()
        self.slam = agibot_gdk.Slam()

        # 读取地图导航点
        print("正在从地图读取导航点...")
        self.waypoints, self.map_id = load_map_waypoints()
        print(f"地图 id={self.map_id}，共读到 {len(self.waypoints)} 个导航点")

        # 合并 waypoints.json（补充自定义点）
        json_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "waypoints.json")
        if os.path.exists(json_file):
            with open(json_file, encoding="utf-8") as f:
                extra = json.load(f)
            for k, v in extra.items():
                if k not in self.waypoints:
                    v["source"] = "json"
                    self.waypoints[k] = v
            print(f"已合并 waypoints.json，共 {len(self.waypoints)} 个导航点")

        self._wp_list = list(self.waypoints.keys())

        signal.signal(signal.SIGINT,  self._on_exit)
        signal.signal(signal.SIGTERM, self._on_exit)

    def _on_exit(self, *_):
        print("\n收到退出信号，取消导航任务...")
        self._cancel()
        sys.exit(0)

    def _cancel(self):
        try:
            ts = self.pnc.get_task_state()
            if ts.state not in DONE_STATES:
                self.pnc.cancel_task(ts.id)
                time.sleep(0.5)
        except Exception:
            pass

    def _task_state(self):
        ts = self.pnc.get_task_state()
        return ts.state, ts.id, ts.message

    def _speed(self):
        try:
            odom = self.slam.get_odom_info()
            return (odom.velocity.x**2 + odom.velocity.y**2) ** 0.5
        except Exception:
            return 0.0

    def list_waypoints(self):
        print(f"\n{'─'*60}")
        print(f"{'索引':<6} {'名称':<10} {'来源':<6} {'位置 (x, y)'}")
        print(f"{'─'*60}")
        for i, name in enumerate(self._wp_list):
            wp = self.waypoints[name]
            pos = wp["position"]
            src = wp.get("source", "?")
            print(f"[{i}]    {name:<10} {src:<6} ({pos[0]:.4f}, {pos[1]:.4f})")
        print(f"{'─'*60}")
        print(f"共 {len(self.waypoints)} 个导航点\n")

    def resolve(self, key: str):
        # 先尝试按索引（纯数字且在范围内）
        try:
            idx = int(key)
            if 0 <= idx < len(self._wp_list):
                return self._wp_list[idx]
        except ValueError:
            pass
        # 再按名称查找
        if key in self.waypoints:
            return key
        return None

    def go(self, name: str, timeout: float = 120.0) -> bool:
        if name not in self.waypoints:
            print(f"❌ 找不到导航点: '{name}'")
            return False

        wp  = self.waypoints[name]
        pos = wp["position"]
        ori = wp["orientation"]
        src = wp.get("source", "?")

        # 取消旧任务
        state, _, _ = self._task_state()
        if state not in DONE_STATES and state != 0:
            print(f"   取消旧任务...")
            self._cancel()
            time.sleep(0.5)

        req = agibot_gdk.NaviReq()
        req.target.position.x    = pos[0]
        req.target.position.y    = pos[1]
        req.target.position.z    = pos[2]
        req.target.orientation.x = ori[0]
        req.target.orientation.y = ori[1]
        req.target.orientation.z = ori[2]
        req.target.orientation.w = ori[3]

        print(f"\n🚀 导航到: '{name}' [{src}]")
        print(f"   目标: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

        try:
            self.pnc.normal_navi(req)
        except Exception as e:
            print(f"❌ 发送失败: {e}")
            return False

        # 等待任务启动
        print("   等待启动...", end="", flush=True)
        started = False
        for _ in range(20):
            time.sleep(0.5)
            state, _, msg = self._task_state()
            if state == 2:
                started = True
                print(" 已启动")
                break
            if state in DONE_STATES:
                break

        if not started:
            state, _, msg = self._task_state()
            if state == 9:
                print(f"\n✅ '{name}' 已到达（目标点在当前位置附近）")
                return True
            print(f"\n❌ 任务未能启动: {S.get(state, state)}  {msg}")
            return False

        # 等待导航完成
        start = time.time()
        while time.time() - start < timeout:
            time.sleep(0.5)
            state, _, msg = self._task_state()
            elapsed = time.time() - start
            print(f"\r   {S.get(state, state)}... {elapsed:.0f}s/{timeout:.0f}s", end="", flush=True)

            if state == 9:
                print("\n   等待停止...", end="", flush=True)
                for _ in range(20):
                    time.sleep(0.3)
                    if self._speed() < 0.05:
                        break
                print()
                print(f"✅ 已到达 '{name}'！耗时 {time.time()-start:.1f}s")
                return True

            if state in {7, 8}:
                print(f"\n❌ 导航失败: {S.get(state, state)}  {msg}")
                return False

        print(f"\n⏰ 超时，取消任务")
        self._cancel()
        return False

    def go_sequence(self, names: list, timeout: float = 120.0) -> bool:
        print(f"\n📍 序列导航: {' → '.join(names)}")
        for i, name in enumerate(names):
            print(f"\n[{i+1}/{len(names)}]", end=" ")
            if not self.go(name, timeout):
                print(f"⛔ 在 '{name}' 失败，停止")
                return False
            if i < len(names) - 1:
                time.sleep(1.0)
        print("\n🎉 全部完成！")
        return True


def main():
    ap = argparse.ArgumentParser(
        description="G2导航脚本 —— 自动读取地图导航点",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python3 navi.py --list          列出地图中所有导航点
  python3 navi.py --wp 0          导航到第0个导航点
  python3 navi.py --wp 1          导航到第1个导航点
  python3 navi.py --seq 0 1       依次导航到多个点
        """
    )
    ap.add_argument("--wp",      type=str, default=None)
    ap.add_argument("--seq",     type=str, nargs="+")
    ap.add_argument("--list",    action="store_true")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    nav = Navigator()

    if args.list:
        nav.list_waypoints()
        return

    if args.wp:
        name = nav.resolve(args.wp)
        if not name:
            print(f"❌ 找不到导航点: '{args.wp}'")
            nav.list_waypoints()
            sys.exit(1)
        ok = nav.go(name, args.timeout)
        sys.exit(0 if ok else 1)

    if args.seq:
        names = []
        for k in args.seq:
            n = nav.resolve(k)
            if not n:
                print(f"❌ 找不到导航点: '{k}'")
                sys.exit(1)
            names.append(n)
        ok = nav.go_sequence(names, args.timeout)
        sys.exit(0 if ok else 1)

    nav.list_waypoints()
    print("用法: python3 navi.py --wp <索引或名称>")


if __name__ == "__main__":
    main()