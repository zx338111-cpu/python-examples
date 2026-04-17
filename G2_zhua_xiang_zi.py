from g2_robot_controller import G2RobotController
from robot_controller import RobotController
import time

# ── 第一段：机械臂抓取 ─────────────────────
ctrl = G2RobotController()
ctrl.init()

try:
    ctrl.get_waist_positions()
    ctrl.record_waist_point("Home")

    ctrl.open_gripper()
    time.sleep(1.0)

    ctrl.move_waist_to_point("home")
    time.sleep(0.8)

    ctrl.move_both_arms_ee_to_point("A1", duration_s=2.0)
    time.sleep(0.8)

    ctrl.move_both_arms_ee_to_point("A2", duration_s=2.0)
    time.sleep(1.0)

    ctrl.close_gripper()
    time.sleep(1.0)

    ctrl.move_both_arms_ee_to_point("A3", duration_s=2.0)
    time.sleep(0.8)

    ctrl.move_both_arms_ee_to_point("A4", duration_s=2.0)
    time.sleep(0.8)

    ctrl.move_both_arms_ee_to_point("A5", duration_s=2.0)
    time.sleep(0.8)

    ctrl.move_both_arms_ee_to_point("A6", duration_s=2.0)
    time.sleep(0.8)

    ctrl.move_waist_to_point("Home_1")
    time.sleep(0.8)

finally:
    ctrl.release()

time.sleep(1.0)

# ── 第二段：底盘前进 ─────────────────────
robot = RobotController()
try:
    robot.move_forward(4.0)
finally:
    robot.release()

time.sleep(1.0)

# ── 第三段：机械臂放置 ─────────────────────
ctrl = G2RobotController()
ctrl.init()

try:
    ctrl.move_waist_to_point("home")
    time.sleep(0.8)

    ctrl.move_both_arms_ee_to_point("A1", duration_s=2.0)
    time.sleep(0.8)

    ctrl.move_both_arms_ee_to_point("A2", duration_s=2.0)
    time.sleep(1.0)

    ctrl.open_gripper()
    time.sleep(1.0)

    ctrl.move_both_arms_ee_to_point("A3", duration_s=2.0)
    time.sleep(0.8)

    ctrl.move_both_arms_ee_to_point("A4", duration_s=2.0)
    time.sleep(0.8)

    ctrl.move_waist_to_point("Home_1")
    time.sleep(0.8)

finally:
    ctrl.release()

time.sleep(1.0)

# ── 第四段：底盘后退 ─────────────────────
robot = RobotController()
try:
    robot.go(0) 
finally:
    robot.release()