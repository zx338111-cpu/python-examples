#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from g2_teach_robot_class import G2TeachRobot

WAIST_JOINTS = [
    "idx01_body_joint1",
    "idx02_body_joint2",
    "idx03_body_joint3",
    "idx04_body_joint4",
    "idx05_body_joint5",
]


def waist_dict_to_list(waist_positions):
    return [float(waist_positions[name]) for name in WAIST_JOINTS]


def move_waist_to_named_joint_point(
    robot,
    point_name,
    step_rad=0.06,
    settle_tol=0.02,
    settle_timeout_s=2.0,
    poll_dt=0.05,
):
    """
    腰部按较大步长分段移动，但不是盲目 sleep，
    而是每一段都等待“接近到位”后再发下一段。

    参数说明：
    - step_rad: 每段最大步长，建议 0.06
    - settle_tol: 认为“这一段到位”的最大误差(rad)
    - settle_timeout_s: 单段等待到位超时
    - poll_dt: 轮询当前腰部位置的周期
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
    print(
        f"[MOVE][WAIST] -> {point_name}, "
        f"max_delta={max_delta:.4f} rad, "
        f"segments={segments}, "
        f"step_rad={step_rad}, settle_tol={settle_tol}"
    )

    for i in range(1, segments + 1):
        alpha = i / segments
        interm = [c + (t - c) * alpha for c, t in zip(current_list, target_list)]

        print(f"[MOVE][WAIST] segment {i}/{segments}")
        robot.move_waist_positions(interm)

        # 关键：等这一段真的接近到位，再发下一段
        t0 = time.time()
        while True:
            now_pos = robot.get_waist_joint_positions()
            now_list = [now_pos[name] for name in WAIST_JOINTS]
            err = max(abs(a - b) for a, b in zip(now_list, interm))

            if err <= settle_tol:
                break

            if time.time() - t0 > settle_timeout_s:
                raise RuntimeError(
                    f"Waist segment {i}/{segments} 未在 {settle_timeout_s:.1f}s 内到位，"
                    f"当前最大误差 {err:.4f} rad"
                )

            time.sleep(poll_dt)

    print(f"[OK] 腰部已移动到点位 {point_name}")


def goto_named_joint_pose(
    robot,
    point_name,
    arm_duration_s=6.0,
    move_waist_first=True,
    waist_step_rad=0.06,
):
    point = robot.get_point(point_name)
    if point is None:
        raise RuntimeError(f"未找到点位: {point_name}")
    if point.get("type") != "joint":
        raise RuntimeError(f"{point_name} 不是关节点")

    print(f"[RUN] 执行命名关节点: {point_name}")

    if move_waist_first:
        move_waist_to_named_joint_point(
            robot,
            point_name,
            step_rad=waist_step_rad,
            settle_tol=0.02,
            settle_timeout_s=2.0,
            poll_dt=0.05,
        )
        robot.goto_joint_point(point_name, duration_s=arm_duration_s)
    else:
        robot.goto_joint_point(point_name, duration_s=arm_duration_s)
        move_waist_to_named_joint_point(
            robot,
            point_name,
            step_rad=waist_step_rad,
            settle_tol=0.02,
            settle_timeout_s=2.0,
            poll_dt=0.05,
        )

    print(f"[OK] 命名关节点执行完成: {point_name}")


def safe_goto_named_joint_pose(
    robot,
    point_name,
    arm_duration_s=6.0,
    move_waist_first=True,
    waist_step_rad=0.06,
    retries=3,
    retry_wait_s=1.0,
):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            goto_named_joint_pose(
                robot,
                point_name,
                arm_duration_s=arm_duration_s,
                move_waist_first=move_waist_first,
                waist_step_rad=waist_step_rad,
            )
            return
        except RuntimeError as e:
            last_err = e
            print(f"[WARN] 第 {attempt}/{retries} 次执行 {point_name} 失败: {e}")
            if attempt < retries:
                print(f"[INFO] 等待 {retry_wait_s:.1f}s 后重试...")
                time.sleep(retry_wait_s)
            else:
                raise
    if last_err is not None:
        raise last_err


def safe_goto_joint_point(robot, point_name, duration_s=4.0, retries=3, retry_wait_s=1.0):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            robot.goto_joint_point(point_name, duration_s=duration_s)
            return
        except RuntimeError as e:
            last_err = e
            print(f"[WARN] 第 {attempt}/{retries} 次执行关节点 {point_name} 失败: {e}")
            if attempt < retries:
                print(f"[INFO] 等待 {retry_wait_s:.1f}s 后重试...")
                time.sleep(retry_wait_s)
            else:
                raise
    if last_err is not None:
        raise last_err


def safe_goto_linear_point(robot, point_name, duration_s=2.5, retries=2, retry_wait_s=0.5):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            robot.goto_linear_point(point_name, duration_s=duration_s)
            return
        except RuntimeError as e:
            last_err = e
            print(f"[WARN] 第 {attempt}/{retries} 次执行直线点 {point_name} 失败: {e}")
            if attempt < retries:
                print(f"[INFO] 等待 {retry_wait_s:.1f}s 后重试...")
                time.sleep(retry_wait_s)
            else:
                raise
    if last_err is not None:
        raise last_err


def main():
    points_file = "g2_points.json"

    home_name = "J_HOME_BOTH"
    grasp_name = "J_PICK_BOTH"
    lift_name = "L_LIFT_BOTH"
    place_ready_name = "J_PLACE_READY_BOTH"
    place_name = "L_PLACE_BOTH"

    close_gripper_after_pick = False
    open_gripper_after_place = False
    return_home = True

    # 如果运行前机器人已经在 HOME，保持 True 可以跳过开头那次回 HOME
    assume_current_is_home = True

    with G2TeachRobot(points_file=points_file) as robot:
        print("========== 当前状态 ==========")
        print(robot.format_status())
        print("=============================\n")

        if not assume_current_is_home:
            safe_goto_named_joint_pose(
                robot,
                home_name,
                arm_duration_s=6.0,
                move_waist_first=False,
                waist_step_rad=0.06,
                retries=3,
                retry_wait_s=1.0,
            )
            time.sleep(0.3)

        # 1) 到抓取位：先腰部，再双臂
        safe_goto_named_joint_pose(
            robot,
            grasp_name,
            arm_duration_s=6.0,
            move_waist_first=True,
            waist_step_rad=0.06,
            retries=3,
            retry_wait_s=1.0,
        )
        time.sleep(0.3)

        # 2) 夹爪闭合（第一次建议先关闭自动）
        if close_gripper_after_pick:
            robot.close_gripper("both")
            time.sleep(0.3)

        # 3) 抬升
        safe_goto_linear_point(
            robot,
            lift_name,
            duration_s=2.5,
            retries=2,
            retry_wait_s=0.5,
        )
        time.sleep(0.3)

        # 4) 到放置预备位
        safe_goto_joint_point(
            robot,
            place_ready_name,
            duration_s=4.0,
            retries=3,
            retry_wait_s=1.0,
        )
        time.sleep(0.3)

        # 5) 到放置点
        safe_goto_linear_point(
            robot,
            place_name,
            duration_s=2.5,
            retries=2,
            retry_wait_s=0.5,
        )
        time.sleep(0.3)

        # 6) 打开夹爪（第一次建议先关闭自动）
        if open_gripper_after_place:
            robot.open_gripper("both")
            time.sleep(0.3)

        # 7) 回 HOME
        if return_home:
            safe_goto_named_joint_pose(
                robot,
                home_name,
                arm_duration_s=6.0,
                move_waist_first=True,
                waist_step_rad=0.06,
                retries=3,
                retry_wait_s=1.0,
            )

        print("\n========== 流程执行完成 ==========")
        print(robot.format_status())
        print("==================================")


if __name__ == "__main__":
    main()
