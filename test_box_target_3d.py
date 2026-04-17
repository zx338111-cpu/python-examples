import time
import numpy as np

from g2_robot_controller import G2RobotController
from g2_visual_perception import G2VisualPerception

MODEL_PATH = "best.pt"
TARGET_CLASS = "box"
CALIB_PATH = "calibration.yaml"

ctrl = G2RobotController()
vp = G2VisualPerception(config_path="vision_config.yaml")

try:
    assert ctrl.init(), "控制器初始化失败"

    # 这里改成你已经验证过的“看箱子姿态”
    # 如果你已经把 look_at_direction 映射改对了，就直接这样
    ctrl.look_at_direction(pitch=0.15, velocity=0.15, wait=True)
    time.sleep(0.5)

    assert vp.init_cameras(), "相机初始化失败"
    time.sleep(1.5)

    ok = vp.load_calibration(CALIB_PATH)
    print("load_calibration:", ok)

    ok = vp.load_model(
        MODEL_PATH,
        model_type="yolo",
        target_classes=[TARGET_CLASS],
        conf=0.5,
    )
    print("load_model:", ok)
    if not ok:
        raise RuntimeError("模型加载失败")

    # 1) 先做 2D 检测
    dets = vp.detect_objects(cam_id="kHeadColor")
    print("detections:", len(dets))
    if not dets:
        raise RuntimeError("没有检测到箱子")

    # 取最高置信度目标
    best = max(dets, key=lambda d: d.confidence)
    print("best_2d:",
          "class=", best.object_id,
          "conf=", best.confidence,
          "bbox=", best.bbox)

    # 2) 测相机坐标系下的 3D
    det3d = vp.estimate_3d_pose(
        best,
        color_cam_id="kHeadColor",
        depth_cam_id="kHeadDepth",
    )

    if det3d is None:
        raise RuntimeError("3D 估计失败")

    print("position_cam (m):", det3d.position_cam)

    # 3) 如果标定加载成功，再测 base_link
    if ok:
        result = vp.transform_cam_to_base(
            det3d.position_cam,
            det3d.orientation_cam,
        )
        print("transform result:", result)
        if result is not None:
            pos_base, ori_base = result
            print("position_base (m):", pos_base)
            print("orientation_base:\n", ori_base)

    # 4) 最后再测一站式接口
    if ok:
        target = vp.get_execution_target(
            object_id="box",
            cam_id="kHeadColor",
            depth_cam_id="kHeadDepth",
            grasp_offset=np.array([0.0, 0.0, 0.0], dtype=float),
        )
        print("target.is_valid:", target.is_valid)
        print("target.message:", target.message)
        if target.is_valid:
            print("target.position:", target.position)
            print("target.quaternion:", target.quaternion)

finally:
    vp.safe_stop()
    ctrl.release()