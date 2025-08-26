#!/usr/bin/env python3
import agibot_gdk
import time
import numpy as np
import cv2
import json
import sys
import threading
import queue
import select
import os

sys.path.insert(0, '/home/agi/bottle_task')
from detect_bottle import detect_bottle_depth # type: ignore


# ---------- 语音命令映射 ----------
# 语音关键词 -> 对应的返回值（与键盘输入一致）
VOICE_CMD_MAP = {
    "扫描瓶子":'',
    "标记此位置": '',      # 相当于 Enter
    "OK":'y',
    "是": 'y',
    "对": 'y',
    "确认": 'y',
    "NO":'n',
    "否": 'n',
    "不对": 'n',
    "取消": 'n',
    "完成": 'q',
    "退出": 'q',
    "结束": 'q'
}

# ---------- 全局变量 ----------
voice_queue = queue.Queue()          # 存放解析后的有效命令
last_asr_text = ""                   # 去重用
interaction = None                   # Interaction对象
asr_handler = None                   # 回调处理器
stop_asr = threading.Event()         # 通知ASR线程退出

# ---------- GDK初始化 ----------
agibot_gdk.gdk_init()
robot = agibot_gdk.Robot()
tf    = agibot_gdk.TF()
cam   = agibot_gdk.Camera()
time.sleep(3)

# ---------- 语音识别回调 ----------
class ASRHandler:
    def __init__(self):
        self.should_exit = False

    def callback(self, text):
        global last_asr_text
        # 过滤空文本和重复文本
        if not isinstance(text, str) or not text.strip():
            return
        if text == last_asr_text:
            return
        last_asr_text = text

        print(f"\n🎤 识别到: {text}")
        # 检查是否包含命令关键词
        for keyword, cmd in VOICE_CMD_MAP.items():
            if keyword in text:
                voice_queue.put(cmd)
                break

def init_voice():
    """初始化语音识别并开启连续监听"""
    global interaction, asr_handler
    interaction = agibot_gdk.Interaction()
    # 开启通话模式（持续语音识别）
    interaction.set_call_mode(True)
    # 注册回调
    asr_handler = ASRHandler()
    interaction.register_callback("get_asr_text", asr_handler.callback)
    print("语音识别已启动，您可以随时使用语音命令。")

def close_voice():
    """关闭语音识别"""
    global interaction, asr_handler
    if interaction:
        try:
            interaction.set_call_mode(False)
            interaction.unregister_callback("get_asr_text")
        except:
            pass
        interaction = None

# ---------- 统一的输入函数（支持键盘和语音）----------
def voice_input(prompt, valid_options=None):
    """
    打印提示，等待用户输入（键盘或语音）。
    valid_options: 允许的返回值列表，如 ['', 'y', 'n', 'q']。
                   若为None，则任何输入均接受（不含空字符串）。
    返回: 用户输入的字符串（去除首尾空白，小写化）。
    """
    print(prompt, end='', flush=True)
    # 如果valid_options为None，则接受任何非空输入
    accept_any = (valid_options is None)
    valid_set = set(valid_options) if valid_options else set()

    # 循环等待有效输入
    while True:
        # 1. 检查键盘输入（非阻塞）
        if select.select([sys.stdin], [], [], 0.1)[0]:
            line = sys.stdin.readline().strip().lower()
            if accept_any or line in valid_set:
                # 键盘输入有效
                print()  # 换行
                return line
            else:
                print(f"\n[无效输入，请重试。允许的值: {valid_options}]", end='', flush=True)

        # 2. 检查语音队列
        try:
            cmd = voice_queue.get_nowait()
            if accept_any or cmd in valid_set:
                # 语音命令有效
                print(f"\n[语音命令: {cmd}]")  # 显示命令
                return cmd
            else:
                # 语音命令不在允许列表中，忽略并继续等待
                pass
        except queue.Empty:
            pass

        # 短暂休眠避免CPU空转
        time.sleep(0.05)

# ---------- 原有功能函数 ----------
def move_head_down():
    req = agibot_gdk.JointControlReq()
    req.life_time = 2.0
    req.joint_names      = ["idx11_head_joint1","idx12_head_joint2","idx13_head_joint3"]
    req.joint_positions  = [0.0, 0.2, 0.0]
    req.joint_velocities = [0.3, 0.3, 0.3]
    try:
        robot.joint_control_request(req)
    except:
        pass

def get_arm_xy():
    t = tf.get_tf_from_base_link("arm_l_end_link")
    return t.translation.x, t.translation.y

def get_cam_3d(u, v, depth_mm):
    intr = cam.get_camera_intrinsic(agibot_gdk.CameraType.kHeadDepth)
    fx, fy = intr.intrinsic[0], intr.intrinsic[1]
    cx, cy = intr.intrinsic[2], intr.intrinsic[3]
    z = depth_mm / 1000.0
    return [(u - cx)*z/fx, (v - cy)*z/fy, z]

# ---------- 主程序 ----------
def main():
    global stop_asr
    try:
        # 启动语音识别
        init_voice()

        print("=" * 55)
        print("相机标定（先检测后对准）")
        print("=" * 55)
        print("""
每个标定点操作步骤：
  第1步: 把瓶子放好，手臂移开（别挡住相机）
  第2步: 说“扫描瓶子”或按Enter → 相机检测瓶子位置
  第3步: 确认检测正确后，再把手臂移到瓶子正上方
  第4步: 说“标记此位置”或按Enter → 记录手臂坐标

目标覆盖: x=0.60~0.88  y=-0.05~0.45  每次移动>15cm

语音命令说明：
  - “标记此位置” /“扫描瓶子” = 确认/继续
  - “是” / “对”    = 输入 y
  - “否” / “不对”  = 输入 n
  - “完成” / “退出” = 结束标定
""")

        move_head_down()
        time.sleep(2)

        cam_points  = []
        base_points = []
        point_num   = 0

        while True:
            point_num += 1
            print(f"\n{'='*45}")
            print(f"标定点 {point_num}  已记录{len(cam_points)}个")
            if base_points:
                xs = [p[0] for p in base_points]
                ys = [p[1] for p in base_points]
                print(f"已覆盖: x={min(xs):.2f}~{max(xs):.2f}  y={min(ys):.2f}~{max(ys):.2f}")

            print("\n【第1步】把瓶子放好，手臂移开别挡住相机")
            choice = voice_input("准备好后说“扫描瓶子”或按Enter检测瓶子，说“完成”结束标定: ",
                                 valid_options=['', 'q'])  # 允许空字符串（Enter）和 q
            if choice == 'q':
                if len(cam_points) < 5:
                    confirm = voice_input(f"只有{len(cam_points)}个点，确定完成？(y/n): ", valid_options=['y', 'n'])
                    if confirm != 'y':
                        continue
                break

            # 检测瓶子
            print("  检测瓶子...")
            result = detect_bottle_depth(cam, save_debug=True)
            if result is None:
                print("  ❌ 未检测到瓶子，请调整位置重试")
                point_num -= 1
                continue

            u, v, depth_mm = result
            cam_pt = get_cam_3d(u, v, depth_mm)
            print(f"  像素=({u},{v})  深度={depth_mm:.0f}mm")
            print(f"  相机坐标=({cam_pt[0]:.3f},{cam_pt[1]:.3f},{cam_pt[2]:.3f})")
            print(f"  查看: http://10.20.15.199:8888/debug_bottle.jpg")

            ans = voice_input("  红点在瓶子上吗？(y/n): ", valid_options=['y', 'n'])
            if ans != 'y':
                print("  ❌ 跳过此点")
                point_num -= 1
                continue

            print("\n【第2步】现在把手臂末端移到瓶子正上方（XY对齐）")
            voice_input("  对准后说“标记此位置”或按Enter记录手臂坐标...", valid_options=[''])

            arm_x, arm_y = get_arm_xy()
            print(f"  手臂: x={arm_x:.4f}  y={arm_y:.4f}")

            # 检查距离
            if base_points:
                min_d = min(abs(arm_x-p[0])+abs(arm_y-p[1]) for p in base_points)
                if min_d < 0.12:
                    print(f"  ⚠️  与已有点太近({min_d*100:.1f}cm)，建议换个更远的位置")
                    confirm = voice_input("  还是要记录？(y/n): ", valid_options=['y', 'n'])
                    if confirm != 'y':
                        point_num -= 1
                        continue

            cam_points.append(cam_pt)
            base_points.append([arm_x, arm_y, 0.808])
            print(f"  ✅ 记录成功！共{len(cam_points)}个点")

            if len(cam_points) >= 5:
                ans = voice_input("  继续添加更多点？(y/n): ", valid_options=['y', 'n'])
                if ans != 'y':
                    break

        if len(cam_points) < 3:
            print("❌ 点不足")
            return

        # 最小二乘拟合
        print(f"\n使用 {len(cam_points)} 个点拟合变换矩阵...")
        A = np.array([[*p, 1.0] for p in cam_points])
        B = np.array(base_points)
        result_m, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

        print("\n验证误差:")
        max_err = 0
        for i, (cp, bp) in enumerate(zip(cam_points, base_points)):
            pred = result_m[:3].T @ np.array(cp) + result_m[3]
            err_cm = np.linalg.norm(np.array(bp[:2]) - pred[:2]) * 100
            max_err = max(max_err, err_cm)
            status = "✅" if err_cm < 5 else "⚠️ "
            print(f"  {status} 点{i+1}: 真实({bp[0]:.3f},{bp[1]:.3f})"
                  f"  预测({pred[0]:.3f},{pred[1]:.3f})"
                  f"  误差={err_cm:.1f}cm")

        print(f"\n最大误差: {max_err:.1f}cm  {'✅ 标定良好！' if max_err<5 else '⚠️ 误差偏大'}")

        calib = {
            "transform_matrix": result_m.tolist(),
            "cam_points":  cam_points,
            "base_points": [p[:2] for p in base_points],
            "max_error_cm": round(max_err, 2),
        }
        with open("/home/agi/bottle_task/cam_calib.json", "w") as f:
            json.dump(calib, f, indent=2)
        print("✅ 保存到 cam_calib.json")

    except KeyboardInterrupt:
        print("\n用户中断程序")
    finally:
        # 清理资源
        close_voice()
        cam.close_camera()
        agibot_gdk.gdk_release()
        print("程序退出")

if __name__ == "__main__":
    main()