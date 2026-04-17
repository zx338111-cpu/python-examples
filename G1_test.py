import time
from g2_robot_controller import G2RobotController

ctrl = G2RobotController()
assert ctrl.init()

def test_head_axis(name, positions, stay=1.5):
    print(f"\n=== 测试 {name}: {positions} ===")
    ctrl.move_head([0.0, 0.0, 0.0], [0.15, 0.15, 0.15], wait=True)
    time.sleep(1.0)

    ctrl.move_head(positions, [0.15, 0.15, 0.15], wait=True)
    time.sleep(stay)

    ctrl.move_head([0.0, 0.0, 0.0], [0.15, 0.15, 0.15], wait=True)
    time.sleep(1.0)

try:
    # joint1
    test_head_axis("joint1 正方向", [0.10, 0.0, 0.0])
    test_head_axis("joint1 负方向", [-0.10, 0.0, 0.0])

    # joint2
    test_head_axis("joint2 正方向", [0.0, 0.10, 0.0])
    test_head_axis("joint2 负方向", [0.0, -0.10, 0.0])

    # joint3
    test_head_axis("joint3 正方向", [0.0, 0.0, 0.10])
    test_head_axis("joint3 负方向", [0.0, 0.0, -0.10])

finally:
    ctrl.move_head([0.0, 0.0, 0.0], [0.15, 0.15, 0.15], wait=True)
    ctrl.release()