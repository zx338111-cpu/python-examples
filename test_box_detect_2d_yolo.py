import time
import cv2

from g2_robot_controller import G2RobotController
from g2_visual_perception import G2VisualPerception

MODEL_PATH = "best.pt"
TARGET_CLASS = "box"

ctrl = G2RobotController()
vp = G2VisualPerception(config_path="vision_config.yaml")

try:
    assert ctrl.init(), "控制器初始化失败"

    # 低头看箱子
    ctrl.look_at_direction(pitch=0.5, velocity=0.15, wait=True)
    time.sleep(0.5)

    # 初始化相机
    assert vp.init_cameras(), "相机初始化失败"
    time.sleep(1.5)

    # 加载本地模型
    ok = vp.load_model(
        MODEL_PATH,
        model_type="yolo",
        target_classes=[TARGET_CLASS],
        conf=0.5,
    )
    print("load_model:", ok)
    if not ok:
        raise RuntimeError("模型加载失败")

    # 跑检测
    dets = vp.detect_objects(cam_id="kHeadColor")
    print("detections:", len(dets))

    frame_data = vp.get_latest_frame("kHeadColor")
    if frame_data is None:
        raise RuntimeError("没有拿到 kHeadColor 图像")

    image, ts = frame_data

    for i, d in enumerate(dets):
        x1, y1, x2, y2 = d.bbox
        print(f"[{i}] class={d.object_id}, conf={d.confidence:.3f}, bbox={d.bbox}")

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{d.object_id}:{d.confidence:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.imwrite("debug_detect_2d_yolo.png", image)
    print("已保存 debug_detect_2d_yolo.png")

finally:
    vp.safe_stop()
    ctrl.release()