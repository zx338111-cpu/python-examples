# import time
# import cv2

# from g2_robot_controller import G2RobotController
# from g2_visual_perception import G2VisualPerception

# ctrl = G2RobotController()
# vp = G2VisualPerception(config_path="vision_config.yaml")

# # try:
# #     assert ctrl.init(), "控制器初始化失败"

# #     # 先低头看箱子
# #     ctrl.look_at_direction(pitch=0.5, velocity=0.15, wait=True)
# #     time.sleep(0.5)

# #     # 初始化视觉相机
# #     ok = vp.init_cameras()
# #     print("init_cameras:", ok)
# #     time.sleep(1.0)

# #     frame_data = vp.get_latest_frame("kHeadColor")
# #     if frame_data is None:
# #         raise RuntimeError("没有拿到 kHeadColor 图像")

# #     image, ts = frame_data
# #     print("timestamp_ns:", ts)
# #     print("image shape:", image.shape)

# #     cv2.imwrite("debug_head_color.png", image)
# #     print("已保存 debug_head_color.png")

# # finally:
# #     vp.safe_stop()
# #     ctrl.release()

# frame_data = vp.get_latest_frame("kHeadDepth")
# if frame_data is None:
#     raise RuntimeError("没有拿到 kHeadDepth 图像")

# depth, ts = frame_data
# print("depth timestamp_ns:", ts)
# print("depth shape:", depth.shape)
# print("depth dtype:", depth.dtype)
# print("depth min/max:", depth.min(), depth.max())



import time
import cv2

from g2_robot_controller import G2RobotController
from g2_visual_perception import G2VisualPerception

ctrl = G2RobotController()
vp = G2VisualPerception(config_path="vision_config.yaml")

try:
    assert ctrl.init(), "控制器初始化失败"

    # 先低头看箱子
    ctrl.look_at_direction(pitch=0.5, velocity=0.15, wait=True)
    time.sleep(0.5)

    # 初始化视觉相机
    ok = vp.init_cameras()
    print("init_cameras:", ok)

    # 稍微等一下相机出流
    time.sleep(2.0)

    # ---------- 先测彩色图 ----------
    color_data = vp.get_latest_frame("kHeadColor")
    if color_data is None:
        raise RuntimeError("没有拿到 kHeadColor 图像")

    color, ts = color_data
    print("color timestamp_ns:", ts)
    print("color shape:", color.shape)

    cv2.imwrite("debug_head_color.png", color)
    print("已保存 debug_head_color.png")

    # ---------- 再测深度图 ----------
    depth_data = vp.get_latest_frame("kHeadDepth")
    if depth_data is None:
        raise RuntimeError("没有拿到 kHeadDepth 图像")

    depth, ts = depth_data
    print("depth timestamp_ns:", ts)
    print("depth shape:", depth.shape)
    print("depth dtype:", depth.dtype)
    print("depth min/max:", depth.min(), depth.max())

finally:
    vp.safe_stop()
    ctrl.release()