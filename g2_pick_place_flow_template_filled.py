#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G2 抓取/运输/放置流程模板（已填入你的 HOME 常量）
基于 g2_teach_robot_class.G2TeachRobot

你的流程：
1. 机器人在 HOME（含双臂 + 腰部）
2. 腰部到抓取位置（抓取点里的 waist_positions）
3. 双臂到抓取位置（抓取点里的 joint_positions）
4. 抬升
5. 收手
6. 回腰（回 HOME 腰部）
7. 运输
8. 放置
"""

from __future__ import annotations

import datetime as dt
import time
from typing import Any, Dict, List

from g2_teach_robot_class import G2TeachRobot, WAIST_JOINTS


HOME_NAME = "J_HOME_BOTH"
GRASP_NAME = "J_PICK_BOTH"
LIFT_NAME = "L_LIFT_BOTH"
CARRY_JOINT_NAME = "J_CARRY_BOTH"
CARRY_LINE_NAME = "L_CARRY_BOTH"
PLACE_READY_NAME = "J_PLACE_READY_BOTH"
PLACE_NAME = "L_PLACE_BOTH"

# ===== 这里是你刚刚读出来的 HOME 常量 =====
HOME_ARM_JOINTS: Dict[str, float] = {
    "idx21_arm_l_joint1":  1.570796,
    "idx22_arm_l_joint2": -1.570796,
    "idx23_arm_l_joint3": -1.570796,
    "idx24_arm_l_joint4": -1.570796,
    "idx25_arm_l_joint5":  0.000000,
    "idx26_arm_l_joint6": -0.000000,
    "idx27_arm_l_joint7":  0.000000,
    "idx61_arm_r_joint1": -1.570797,
    "idx62_arm_r_joint2": -1.570796,
    "idx63_arm_r_joint3":  1.570796,
    "idx64_arm_r_joint4": -1.570796,
    "idx65_arm_r_joint5":  0.000000,
    "idx66_arm_r_joint6":  0.000000,
    "idx67_arm_r_joint7": -0.000000,
}

HOME_WAIST_JOINTS: Dict[str, float] = {
    "idx01_body_joint1": -0.698016,
    "idx02_body_joint2":  1.570727,
    "idx03_body_joint3": -0.872522,
    "idx04_body_joint4":  0.000000,
    "idx05_body_joint5":  0.000000,
}

HOME_GRIPPERS: Dict[str, float] = {
    "left": 0.0,
    "right": 0.0,
}


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def waist_dict_to_list(waist_positions: Dict[str, float]) -> List[float]:
    return [float(waist_positions[name]) for name in WAIST_JOINTS]


def print_saved_point_summary(robot: G2TeachRobot, name: str) -> None:
    point = robot.get_point(name)
    if point is None:
        raise RuntimeError(f"未找到点位: {name}")
    print(f"\n=== 点位 {name} ===")
    print(f"type={point.get('type')}, scope={point.get('scope')}, created_at={point.get('created_at')}")
    jp = point.get("joint_positions", {})
    wp = point.get("waist_positions", {})
    if jp:
        print("双臂关节：")
        for k, v in jp.items():
            print(f"  {k}: {float(v):.6f} rad")
    if wp:
        print("腰部关节：")
        for k in WAIST_JOINTS:
            if k in wp:
                print(f"  {k}: {float(wp[k]):.6f} rad")
    print()


def seed_home_point_from_constants(
    robot: G2TeachRobot,
    name: str = HOME_NAME,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    用你已经读出来的 HOME 常量，直接在 points 文件里初始化/覆盖 HOME 点。
    这样不需要机器人当前真的停在 HOME，也能先把 HOME 点写进去。
    """
    exists = robot.get_point(name)
    if exists is not None and not overwrite:
        print(f"[INFO] 点位 {name} 已存在，跳过初始化。")
        return exists

    point = {
        "name": name,
        "type": "joint",
        "scope": "both",
        "created_at": now_iso(),
        "joint_positions": dict(HOME_ARM_JOINTS),
        "waist_positions": dict(HOME_WAIST_JOINTS),
        "grippers": dict(HOME_GRIPPERS),
        "note": "HOME seeded from measured constants",
    }
    robot.points.upsert(point)
    print(f"[OK] 已用常量初始化 HOME 点: {name}")
    print_saved_point_summary(robot, name)
    return point


def save_home_snapshot(robot: G2TeachRobot, name: str = HOME_NAME) -> None:
    """
    如果机器人当前确实处于 HOME，可以用实时状态覆盖保存。
    """
    point = robot.save_joint_point(name, scope="both", note="HOME snapshot: arms + waist + grippers")
    print(f"[OK] 已保存 HOME 点: {name}")
    print_saved_point_summary(robot, name)


def save_grasp_snapshot(robot: G2TeachRobot, name: str = GRASP_NAME) -> None:
    point = robot.save_joint_point(name, scope="both", note="GRASP snapshot: arms + waist + grippers")
    print(f"[OK] 已保存 GRASP 点: {name}")
    print_saved_point_summary(robot, name)


def move_waist_to_named_joint_point(
    robot: G2TeachRobot,
    point_name: str,
    step_rad: float = 0.01,
    pause_s: float = 0.25,
) -> None:
    """
    将腰部逐步移动到某个已保存关节点里的 waist_positions。
    """
    point = robot.get_point(point_name)
    if point is None:
        raise RuntimeError(f"未找到点位: {point_name}")
    waist_positions = point.get("waist_positions")
    if not waist_positions:
        raise RuntimeError(f"点位 {point_name} 不含 waist_positions")

    current = robot.get_waist_joint_positions()
    current_list = [current[name] for name in WAIST_JOINTS]
    target_list = waist_dict_to_list(waist_positions)

    max_delta = max(abs(t - c) for c, t in zip(current_list, target_list))
    if max_delta < 1e-4:
        print(f"[MOVE][WAIST] 已在 {point_name} 腰部目标附近，无需移动")
        return

    segments = max(1, int(max_delta / max(step_rad, 1e-4)))
    print(f"[MOVE][WAIST] -> {point_name}, max_delta={max_delta:.4f} rad, segments={segments}")

    for i in range(1, segments + 1):
        alpha = i / segments
        interm = [c + (t - c) * alpha for c, t in zip(current_list, target_list)]
        robot.move_waist_positions(interm)
        time.sleep(pause_s)

    print(f"[OK] 腰部已移动到点位 {point_name}")


def goto_named_joint_pose(
    robot: G2TeachRobot,
    point_name: str,
    arm_duration_s: float = 6.0,
    move_waist_first: bool = True,
    waist_step_rad: float = 0.01,
) -> None:
    """
    执行“腰部 + 双臂”的命名关节点。
    默认顺序：
    1. 腰部先到位
    2. 双臂再到位
    """
    point = robot.get_point(point_name)
    if point is None:
        raise RuntimeError(f"未找到点位: {point_name}")
    if point.get("type") != "joint":
        raise RuntimeError(f"{point_name} 不是关节点")

    print(f"[RUN] 执行命名关节点: {point_name}")
    if move_waist_first:
        move_waist_to_named_joint_point(robot, point_name, step_rad=waist_step_rad)
        robot.goto_joint_point(point_name, duration_s=arm_duration_s)
    else:
        robot.goto_joint_point(point_name, duration_s=arm_duration_s)
        move_waist_to_named_joint_point(robot, point_name, step_rad=waist_step_rad)
    print(f"[OK] 命名关节点执行完成: {point_name}")


def run_pick_carry_place(
    robot: G2TeachRobot,
    home_name: str = HOME_NAME,
    grasp_name: str = GRASP_NAME,
    lift_name: str = LIFT_NAME,
    place_ready_name: str = PLACE_READY_NAME,
    place_name: str = PLACE_NAME,
) -> None:
    print(robot.format_status())

    # 0) 从 HOME 起步
    goto_named_joint_pose(robot, home_name, arm_duration_s=6.0, move_waist_first=False)

    # 1) 腰部到抓取位置 + 双臂到抓取位置
    goto_named_joint_pose(robot, grasp_name, arm_duration_s=6.0, move_waist_first=True)

    # 2) 抬升：优先使用已保存好的抬升直线点
    if robot.get_point(lift_name) is not None:
        robot.goto_linear_point(lift_name, duration_s=2.5)
    else:
        robot.lift_arms(dz=0.10, duration_s=2.0)

    # 3) 收手 + 小步回腰（运输姿态）
    carry_result = robot.make_carry_posture(
        retract_x=0.10,
        lift_z=0.00,
        waist_step_rad=0.01,
        waist_max_iters=3,
        arm_duration_s=2.0,
    )
    print("[INFO] carry posture result:", carry_result)

    # 4) 回腰：回到 HOME 的腰部
    move_waist_to_named_joint_point(robot, home_name, step_rad=0.01)

    # 5) 运输到放置预备位（关节）
    if robot.get_point(place_ready_name) is not None:
        robot.goto_joint_point(place_ready_name, duration_s=4.0)

    # 6) 放置（直线）
    if robot.get_point(place_name) is not None:
        robot.goto_linear_point(place_name, duration_s=2.5)

    print("[OK] 流程执行结束")
    print(robot.format_status())


def demo():
    with G2TeachRobot(points_file="g2_points.json") as robot:
        # 1) 先把 HOME 常量写入点位文件（如果已存在则跳过）
        seed_home_point_from_constants(robot, HOME_NAME, overwrite=False)

        # 2) 你如果此刻机器人真的在 HOME，也可以用实时状态覆盖 HOME 点
        # save_home_snapshot(robot, HOME_NAME)

        # 3) 你如果此刻机器人在抓取位，可以保存 GRASP 点
        # save_grasp_snapshot(robot, GRASP_NAME)

        # 4) 真正跑流程时，取消下一行注释
        # run_pick_carry_place(robot)


if __name__ == "__main__":
    demo()