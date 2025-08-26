#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
record_pose.py  —  读取G2机器人当前位置并保存为导航点

用法：
  source ~/.cache/agibot/app/env.sh
  python3 record_pose.py              # 读取一次当前位置
  python3 record_pose.py --name 客厅  # 指定名称保存
  python3 record_pose.py --loop       # 交互式循环记录多个点
"""

import argparse
import json
import os
import time

import agibot_gdk


SAVE_FILE = "waypoints.json"   # 保存文件名


def load_waypoints() -> dict:
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_waypoints(data: dict):
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ 已保存到 {SAVE_FILE}")


def get_current_pose(slam) -> dict:
    """读取当前位姿，返回字典"""
    odom = slam.get_odom_info()
    pos = odom.pose.pose.position
    ori = odom.pose.pose.orientation
    return {
        "position": [
            round(pos.x, 6),
            round(pos.y, 6),
            round(pos.z, 6)
        ],
        "orientation": [
            round(ori.x, 6),
            round(ori.y, 6),
            round(ori.z, 6),
            round(ori.w, 6)
        ],
        "loc_state":      odom.loc_state,
        "loc_confidence": round(odom.loc_confidence, 4)
    }


def print_pose(name: str, pose: dict):
    pos = pose["position"]
    ori = pose["orientation"]
    print(f"\n{'='*50}")
    print(f"  导航点: {name}")
    print(f"{'='*50}")
    print(f"  位置:   x={pos[0]},  y={pos[1]},  z={pos[2]}")
    print(f"  朝向:  qx={ori[0]}, qy={ori[1]}, qz={ori[2]}, qw={ori[3]}")
    print(f"  定位状态: {pose['loc_state']},  置信度: {pose['loc_confidence']}")
    print(f"{'='*50}")
    if pose["loc_confidence"] < 0.5:
        print("  ⚠️  置信度较低，建议先在Pad上重定位后再记录")


def main():
    ap = argparse.ArgumentParser(description="记录G2机器人当前位置为导航点")
    ap.add_argument("--name", type=str, default=None, help="导航点名称")
    ap.add_argument("--loop", action="store_true",   help="循环记录模式，可连续记录多个点")
    args = ap.parse_args()

    # 初始化
    print("正在连接机器人...")
    agibot_gdk.gdk_init()
    slam = agibot_gdk.Slam()
    time.sleep(1.5)
    print("连接成功\n")

    waypoints = load_waypoints()
    if waypoints:
        print(f"已有导航点文件，包含 {len(waypoints)} 个导航点: {list(waypoints.keys())}")

    try:
        if args.loop:
            # ── 交互式循环记录 ──────────────────────────
            print("进入循环记录模式，输入名称记录当前位置，直接回车退出\n")
            while True:
                name = input("导航点名称（回车退出）: ").strip()
                if not name:
                    break

                pose = get_current_pose(slam)
                print_pose(name, pose)

                confirm = input("确认保存? (y/n, 默认y): ").strip().lower()
                if confirm in ("", "y", "yes"):
                    waypoints[name] = pose
                    save_waypoints(waypoints)
                else:
                    print("已跳过")

        else:
            # ── 单次记录 ────────────────────────────────
            pose = get_current_pose(slam)
            name = args.name or f"点_{len(waypoints)+1}"
            print_pose(name, pose)

            save = input("\n是否保存? (y/n, 默认y): ").strip().lower()
            if save in ("", "y"):
                if not args.name:
                    custom = input(f"导航点名称（默认 '{name}'）: ").strip()
                    if custom:
                        name = custom
                waypoints[name] = pose
                save_waypoints(waypoints)

    except KeyboardInterrupt:
        print("\n退出")
    finally:
        agibot_gdk.gdk_release()

    # 最终汇总
    if waypoints:
        print(f"\n当前 {SAVE_FILE} 共有 {len(waypoints)} 个导航点:")
        for n, p in waypoints.items():
            pos = p["position"]
            print(f"  - {n}: ({pos[0]:.3f}, {pos[1]:.3f})")

        print("\n复制以下内容到 navi_to_waypoint.py 的 WAYPOINTS 字典中：\n")
        print("WAYPOINTS = {")
        for n, p in waypoints.items():
            pos = p["position"]
            ori = p["orientation"]
            print(f'    "{n}": {{')
            print(f'        "position":    {pos},')
            print(f'        "orientation": {ori},')
            print(f'    }},')
        print("}")


if __name__ == "__main__":
    main()