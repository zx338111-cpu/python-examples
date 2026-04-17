#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crab_walk.py — 蟹行横移指定距离

用法：
  python3 crab_walk.py            # 向左横移3米（默认）
  python3 crab_walk.py --dist 1.5 # 横移1.5米
  python3 crab_walk.py --dist -3  # 向右横移3米（负数=右）
  python3 crab_walk.py --speed 0.3 --dist 2.0
"""

import argparse
import time
import agibot_gdk

# 控制参数
CTRL_HZ    = 20       # 控制频率 20Hz
CTRL_DT    = 1.0 / CTRL_HZ


def crab_walk(dist_m: float, speed: float = 0.2):
    """
    蟹行横移
    dist_m > 0 向左，dist_m < 0 向右
    speed: 横向速度 m/s
    """
    agibot_gdk.gdk_init()
    pnc = agibot_gdk.Pnc()
    slam = agibot_gdk.Slam()
    time.sleep(0.8)

    # 申请蟹行控制权
    pnc.request_chassis_control(1)
    time.sleep(0.3)

    # 根据距离和速度计算时间
    direction = 1.0 if dist_m >= 0 else -1.0
    vy = direction * abs(speed)
    duration = abs(dist_m) / abs(speed)

    print(f"蟹行横移: {dist_m:+.2f} 米  速度: {vy:+.2f} m/s  预计时间: {duration:.1f}s")
    print(f"方向: {'← 左' if dist_m >= 0 else '→ 右'}")
    print("开始移动...")

    # 构建速度命令
    twist = agibot_gdk.Twist()
    twist.linear  = agibot_gdk.Vector3()
    twist.angular = agibot_gdk.Vector3()
    twist.linear.x  = 0.0
    twist.linear.y  = vy
    twist.linear.z  = 0.0
    twist.angular.x = 0.0
    twist.angular.y = 0.0
    twist.angular.z = 0.0

    # 用里程计估算已走距离
    try:
        odom0 = slam.get_odom_info()
        y0 = odom0.pose.pose.position.y
        x0 = odom0.pose.pose.position.x
        use_odom = True
    except Exception:
        use_odom = False
        y0 = x0 = 0.0

    start = time.time()
    elapsed = 0.0

    try:
        while elapsed < duration:
            pnc.move_chassis(twist)
            time.sleep(CTRL_DT)
            elapsed = time.time() - start

            # 实时显示进度
            if use_odom:
                try:
                    odom = slam.get_odom_info()
                    moved = ((odom.pose.pose.position.x - x0)**2 +
                             (odom.pose.pose.position.y - y0)**2) ** 0.5
                    print(f"\r  已移动: {moved:.3f}m / {abs(dist_m):.1f}m  "
                          f"用时: {elapsed:.1f}s", end="", flush=True)
                except Exception:
                    print(f"\r  用时: {elapsed:.1f}s / {duration:.1f}s", end="", flush=True)
            else:
                print(f"\r  用时: {elapsed:.1f}s / {duration:.1f}s", end="", flush=True)

    except KeyboardInterrupt:
        print("\n⚠️  用户中断")

    # 停止
    twist.linear.y = 0.0
    for _ in range(5):
        pnc.move_chassis(twist)
        time.sleep(0.05)

    print(f"\n✅ 完成！总用时: {time.time()-start:.1f}s")

    # 取消任务
    try:
        ts = pnc.get_task_state()
        pnc.cancel_task(ts.id)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="蟹行横移脚本")
    ap.add_argument("--dist",  type=float, default=3.0,
                    help="横移距离(米)，正=左，负=右，默认3.0")
    ap.add_argument("--speed", type=float, default=0.2,
                    help="横向速度(m/s)，默认0.2")
    args = ap.parse_args()

    if abs(args.dist) < 0.01:
        print("距离太小，退出")
        return
    if args.speed <= 0 or args.speed > 1.0:
        print("速度需在 0~1.0 m/s 之间")
        return

    crab_walk(args.dist, args.speed)


if __name__ == "__main__":
    main()