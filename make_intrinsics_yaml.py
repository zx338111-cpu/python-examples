import time
import yaml
import agibot_gdk

def camera_intr_to_dict(camera, cam_type):
    intr = camera.get_camera_intrinsic(cam_type)
    shape = camera.get_image_shape(cam_type)

    return {
        "fx": float(intr.intrinsic[0]),
        "fy": float(intr.intrinsic[1]),
        "cx": float(intr.intrinsic[2]),
        "cy": float(intr.intrinsic[3]),
        "width": int(shape[0]),
        "height": int(shape[1]),
        "dist_coeffs": [float(x) for x in intr.distortion],
    }

assert agibot_gdk.gdk_init() == agibot_gdk.GDKRes.kSuccess
camera = agibot_gdk.Camera()
time.sleep(2.0)

data = {
    "intrinsics": {
        "kHeadColor": camera_intr_to_dict(camera, agibot_gdk.CameraType.kHeadColor),
        "kHeadDepth": camera_intr_to_dict(camera, agibot_gdk.CameraType.kHeadDepth),
    },
    "stereo_baseline": 0.065,
}

with open("calibration.yaml", "w", encoding="utf-8") as f:
    yaml.dump(data, f, allow_unicode=True, sort_keys=False)

print("已生成 calibration.yaml")
agibot_gdk.gdk_release()