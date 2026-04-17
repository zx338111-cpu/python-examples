from robot_controller import RobotController
import math
robot = RobotController()

# 导航到地图第0个点
# robot.go(1)



# robot.go(1, high_precision=True)

#robot.crab_walk(-2.0)
# robot.move_forward(5)
# robot.move_forward(-5)



# from robot_controller import RobotController
# import math

# robot = RobotController()

# # 导航到地图第0个点
# robot.go(0)

# # 高精度导航
# robot.go(0, high_precision=True)

# # 蟹行左移1.5米
# robot.crab_walk(1.5)

# # 蟹行右移2米
# robot.crab_walk(-2.0)

# # 前进1米
# robot.move_forward(1.0)

# # 左转90度
# robot.rotate(math.radians(90))

# # 序列导航
# robot.go_sequence([0, 1])




# python3 robot_controller.py --list
# python3 robot_controller.py --wp 0
# python3 robot_controller.py --crab 3.0
# python3 robot_controller.py --crab -3.0
# python3 robot_controller.py --forward 2.0
# python3 robot_controller.py --rotate 90
# python3 robot_controller.py --rotate -45 --speed 0.3




robot.move_forward(1.0)
robot.rotate(0.5)
robot.crab_walk(0.5)

robot.release()
