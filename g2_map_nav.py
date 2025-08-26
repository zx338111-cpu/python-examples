#!/usr/bin/env python3
"""
g2_map_nav.py

功能：
  - 加载已生成的地图 (map.pgm + map.yaml)
  - 设置目标点
  - 规划路径并控制 G2 机器人底盘自动导航
  - 支持动态避障检测
"""

import time
from agibot_gdk import Map, Pnc, Lidar

# -------------------- 配置参数 --------------------
MAP_PGM_PATH = "/home/agi/app/gdk/examples/python/map.pgm"
MAP_YAML_PATH = "/home/agi/app/gdk/examples/python/map.yaml"

# 目标点 (x, y, theta)，单位: 米 / 弧度
GOAL = (2.0, 1.5, 0.0)

# 控制循环周期
CTRL_DT = 0.1  # 秒

# 最大速度
VX_MAX = 0.2    # 前后 m/s
VY_MAX = 0.1    # 左右 m/s
WZ_MAX = 0.3    # 旋转 rad/s

# 避障距离阈值
OBSTACLE_DIST = 0.3  # 米

# -------------------- 初始化 --------------------
print("[INFO] 初始化 G2 SDK...")
pnc = Pnc()
lidar = Lidar()
map_obj = Map()

print("[INFO] 加载地图...")
map_obj.load(MAP_PGM_PATH, MAP_YAML_PATH)

# -------------------- 路径规划 --------------------
print(f"[INFO] 规划路径到目标点 {GOAL}...")
start = map_obj.get_robot_pose()  # 当前机器人位置 (x, y, theta)
path = map_obj.plan(start, GOAL)

if path is None or len(path) == 0:
    print("[ERROR] 无法规划路径，请检查地图或目标点")
    exit(1)

print(f"[INFO] 路径点数: {len(path)}")

# -------------------- 导航循环 --------------------
for idx, waypoint in enumerate(path):
    print(f"[NAV] 前往路径点 {idx+1}/{len(path)}: {waypoint}")
    while True:
        # 获取当前机器人状态
        robot_pose = map_obj.get_robot_pose()
        x, y, theta = robot_pose

        # 计算到 waypoint 的偏差
        dx = waypoint[0] - x
        dy = waypoint[1] - y
        dist = (dx**2 + dy**2)**0.5
        desired_theta = waypoint[2]
        dtheta = desired_theta - theta

        # 检查是否到达当前 waypoint
        if dist < 0.05 and abs(dtheta) < 0.05:
            print(f"[NAV] 到达路径点 {idx+1}")
            break

        # -------------------- 避障 --------------------
        obstacle = lidar.get_closest_obstacle_distance()
        if obstacle is not None and obstacle < OBSTACLE_DIST:
            print(f"[NAV] 避障：前方障碍物距离 {obstacle:.2f} m，停止移动")
            pnc.move_chassis(0.0, 0.0, 0.0)
            time.sleep(CTRL_DT)
            continue

        # -------------------- 控制速度 --------------------
        vx = max(-VX_MAX, min(VX_MAX, dx))
        vy = max(-VY_MAX, min(VY_MAX, dy))
        wz = max(-WZ_MAX, min(WZ_MAX, dtheta))

        # 发送到底盘
        pnc.move_chassis(vx, vy, wz)
        time.sleep(CTRL_DT)

# -------------------- 停止底盘 --------------------
print("[NAV] 到达目标点，停止底盘")
pnc.move_chassis(0.0, 0.0, 0.0)
print("[NAV] 导航完成")