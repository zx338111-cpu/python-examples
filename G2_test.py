from g2_robot_controller import G2RobotController
ctrl = G2RobotController()   # ← 自动从磁盘恢复所有点位
import time

from robot_controller import RobotController
import math
robot = RobotController()
ctrl.init()
ctrl.get_waist_positions()  


# ctrl.record_waist_point("grasp_waist")
# ctrl.record_both_arms_ee_point("grasp_pose")

                # 只读取打印
# ctrl.record_waist_point("Home")    

#ctrl.look_at_direction(pitch=-0.3, wait=True)   #低头
#ctrl.look_at_direction(pitch=0.5, wait=True)

#ctrl.move_waist_to_point("home") 
#ctrl.record_both_arms_point("hou")  
#ctrl.move_both_arms_delta(dz=+0.20)   # 双臂上抬 20cm
#ctrl.move_both_arms_delta(dx=-0.10)   # 双臂向后伸 10cm
#ctrl.record_left_arm_ee_point("1")     #保存左臂末端点位

# # 把当前左右臂末端位置+姿态一起记录，写入磁盘，重启自动恢复
# ctrl.record_both_arms_ee_point("suo_hw1")
# ctrl.record_both_arms_ee_point("suo_hw2")


# #双臂同时走直线到记录时的末端位置（50Hz笛卡尔插值）
# ctrl.move_both_arms_ee_to_point("pai_zhao1", duration_s=2.0)
# ctrl.move_both_arms_ee_to_point("pai_zhao2", duration_s=3.0)


# ctrl.record_waist_point("view_waist")
# ctrl.record_both_arms_ee_point("view_pose")

# ctrl.move_both_arms_delta(dy=+0.01, duration_s=1.0)
# time.sleep(1.0)



# ctrl.move_waist_to_point("home")            # 运动到该点（腰部）
# # 左右臂同时发指令运动，阻塞等待双臂都到位
# ctrl.move_both_arms_to_point("zhua_qu")
# # 方式2：同时到达 —— 自动按行程比例分配速度，两臂同时停
# ctrl.move_both_arms_to_point("zhua_qu", sync_arrival=True)
# # 左右臂同时发指令运动，阻塞等待双臂都到位
# ctrl.move_both_arms_to_point("shang_tai",sync_arrival=True)
# # 左右臂同时发指令运动，阻塞等待双臂都到位
# ctrl.move_both_arms_to_point("hou",sync_arrival=True)

# ctrl.open_gripper()
# time.sleep(1.0)

# ctrl.move_waist_to_point("home")
# time.sleep(0.8)

# ctrl.move_both_arms_ee_to_point("A3", duration_s=2.0)
# time.sleep(0.8)

# ctrl.move_both_arms_ee_to_point("A4", duration_s=2.0)
# time.sleep(1.0)

# ctrl.close_gripper()
# time.sleep(1.0)

# ctrl.move_both_arms_ee_to_point("A3", duration_s=2.0)
# time.sleep(0.8)

# ctrl.move_both_arms_ee_to_point("A4", duration_s=2.0)
# time.sleep(0.8)

# ctrl.move_both_arms_ee_to_point("A1", duration_s=2.0)
# time.sleep(0.8)

# ctrl.move_both_arms_ee_to_point("A2", duration_s=2.0)
# time.sleep(0.8)

# ctrl.move_waist_to_point("Home_1")
# time.sleep(0.8)

# robot.move_forward(4.0)  

# ctrl.move_waist_to_point("home")
# time.sleep(0.8)

# ctrl.move_both_arms_ee_to_point("A1", duration_s=2.0)
# time.sleep(0.8)

# ctrl.move_both_arms_ee_to_point("A2", duration_s=2.0)
# time.sleep(1.0)

# ctrl.open_gripper()
# time.sleep(1.0)


# ctrl.move_both_arms_ee_to_point("A3", duration_s=2.0)
# time.sleep(0.8)

# ctrl.move_both_arms_ee_to_point("A4", duration_s=2.0)
# time.sleep(0.8)


# ctrl.move_waist_to_point("Home_1")
# time.sleep(0.8)



# robot.move_forward(-4.0)  


# robot.release()                 # 结束释放

ctrl.release()


