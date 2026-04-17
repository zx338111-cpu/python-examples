#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G2 腰部诊断脚本 —— 运行前先看清楚机器人当前状态
用法：/usr/bin/python g2_diagnose.py
"""
import time
import agibot_gdk

print("=" * 60)
print("G2 诊断脚本")
print("=" * 60)

# 初始化
if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
    print("❌ GDK 初始化失败"); exit(1)
print("✅ GDK 初始化成功")

robot = agibot_gdk.Robot()
time.sleep(2)

# ── 1. 全身状态 ──────────────────────────────────────────
print("\n【全身状态】")
try:
    status = robot.get_whole_body_status()
    print(f"  腰部错误码   waist_error  : {status['waist_error']}")
    print(f"  头部错误码   neck_error   : {status['neck_error']}")
    print(f"  升降错误码   lift_error   : {status['lift_error']}")
    print(f"  左臂错误码   left_arm_error  : {status['left_arm_error']}")
    print(f"  右臂错误码   right_arm_error : {status['right_arm_error']}")
    print(f"  左臂控制状态 left_arm_control  : {status['left_arm_control']}")
    print(f"  右臂控制状态 right_arm_control : {status['right_arm_control']}")
    print(f"  左臂急停     left_arm_estop  : {status['left_arm_estop']}")
    print(f"  右臂急停     right_arm_estop : {status['right_arm_estop']}")
    if status['waist_error'] == 0:
        print("  ✅ 腰部无错误")
    else:
        print(f"  ❌ 腰部有错误码: {status['waist_error']} (0x{status['waist_error']:04X})")
except Exception as e:
    print(f"  get_whole_body_status 失败: {e}")

# ── 2. 各关节状态（重点看腰部） ──────────────────────────
print("\n【腰部关节详细状态】")
try:
    joint_states = robot.get_joint_states()
    waist_names = [
        "idx01_body_joint1", "idx02_body_joint2", "idx03_body_joint3",
        "idx04_body_joint4", "idx05_body_joint5"
    ]
    for state in joint_states['states']:
        if state['name'] in waist_names:
            ec = state['error_code']
            flag = "✅" if ec == 0 else f"❌ 错误码 0x{ec:04X}"
            print(f"  {state['name']:30s}  pos={state['motor_position']:+.3f} rad  error={flag}")
except Exception as e:
    print(f"  get_joint_states 失败: {e}")

# ── 3. 尝试发一条腰部指令，看具体错误 ───────────────────
print("\n【测试 move_waist_joint（原地不动）】")
try:
    # 先读当前位置
    joint_states = robot.get_joint_states()
    cur = {s['name']: s['motor_position'] for s in joint_states['states']}
    waist_names = [
        "idx01_body_joint1", "idx02_body_joint2", "idx03_body_joint3",
        "idx04_body_joint4", "idx05_body_joint5"
    ]
    cur_pos = [cur.get(nm, 0.0) for nm in waist_names]
    print(f"  当前腰部位置: {[f'{p:+.3f}' for p in cur_pos]}")
    print("  发送原地保持指令（速度极低）...")
    result = robot.move_waist_joint(cur_pos, [0.05] * 5)
    print(f"  返回值: {result}  ← 0 表示成功")
except Exception as e:
    print(f"  ❌ 失败: {e}")

# ── 4. 尝试 move_arm_joint ───────────────────────────────
print("\n【测试 move_arm_joint（双臂原地不动）】")
try:
    joint_states = robot.get_joint_states()
    cur = {s['name']: s['motor_position'] for s in joint_states['states']}
    arm_names = [
        "idx21_arm_l_joint1","idx22_arm_l_joint2","idx23_arm_l_joint3",
        "idx24_arm_l_joint4","idx25_arm_l_joint5","idx26_arm_l_joint6","idx27_arm_l_joint7",
        "idx61_arm_r_joint1","idx62_arm_r_joint2","idx63_arm_r_joint3",
        "idx64_arm_r_joint4","idx65_arm_r_joint5","idx66_arm_r_joint6","idx67_arm_r_joint7",
    ]
    arm_pos = [cur.get(nm, 0.0) for nm in arm_names]
    print(f"  左臂位置: {[f'{p:+.3f}' for p in arm_pos[:7]]}")
    print(f"  右臂位置: {[f'{p:+.3f}' for p in arm_pos[7:]]}")
    print("  发送原地保持指令...")
    result = robot.move_arm_joint(arm_pos, [0.05] * 14)
    print(f"  返回值: {result}  ← 0 表示成功")
except Exception as e:
    print(f"  ❌ 失败: {e}")

agibot_gdk.gdk_release()
print("\n✅ 诊断完成")