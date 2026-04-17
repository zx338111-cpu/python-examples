#!/usr/bin/env python3
"""
box_align.py — 控制机器人与箱子的前后距离
箱子摆正，机器人自动调整到标定距离
"""
import argparse, json, os, time, sys
import cv2, numpy as np
import agibot_gdk

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
CALIB_FILE = os.path.join(os.path.dirname(__file__), "calib.json")
HEAD_CAM       = agibot_gdk.CameraType.kHeadColor
HEAD_LOOK_DOWN = [0.0, 0.0, 0.5]
HEAD_SPEED     = [0.2, 0.2, 0.2]
CONF_THR  = 0.25   # 低阈值提高检测率

TOL_DIST  = 8      # 前后容忍（像素）
VX_BASE   = 0.04   # 前后速度 m/s
DT        = 0.08
MAX_STEPS  = 150
NO_BOX_MAX = 60    # 多帧容忍，避免漏检停止

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--model",     default=MODEL_PATH)
    ap.add_argument("--debug",     action="store_true")
    ap.add_argument("--no-move",   action="store_true")
    return ap.parse_args()

def decode_image(img):
    arr = np.frombuffer(bytes(img.data), dtype=np.uint8)
    if img.encoding in (agibot_gdk.Encoding.JPEG, agibot_gdk.Encoding.PNG):
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    frame = arr.reshape((img.height, img.width, 3))
    try:
        if img.color_format == agibot_gdk.ColorFormat.RGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    except Exception:
        pass
    return frame

def grab_frame(camera):
    for _ in range(3):
        img = camera.get_latest_image(HEAD_CAM, 1000.0)
        if img is not None:
            return decode_image(img)
        time.sleep(0.1)
    return None

def detect_box(model, bgr):
    """检测箱子，返回bh（高度像素），越大=越近"""
    results = model(bgr, conf=CONF_THR, verbose=False)
    best, best_conf = None, 0.0
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                x1,y1,x2,y2 = [float(v) for v in box.xyxy[0]]
                best = (x1,y1,x2,y2,conf)
    if best is None:
        return None
    x1,y1,x2,y2,conf = best
    cx = (x1+x2)/2
    cy = (y1+y2)/2
    bw = x2-x1
    bh = y2-y1
    return cx, cy, bw, bh, conf

def draw_debug(bgr, det, calib, step, bh_smooth):
    vis = bgr.copy()
    h, w = vis.shape[:2]
    bh_ref = calib["bh"]
    # 中心线
    cv2.line(vis,(w//2,0),(w//2,h),(255,100,0),1)
    if det:
        cx,cy,bw,bh,conf = det
        x1=int(cx-bw/2); y1=int(cy-bh/2)
        x2=int(cx+bw/2); y2=int(cy+bh/2)
        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.circle(vis,(int(cx),int(cy)),5,(0,0,255),-1)
        d_dist = bh_smooth - bh_ref
        c = (0,255,0) if abs(d_dist)<TOL_DIST else (0,0,255)
        cv2.putText(vis,
            f"bh={bh:.0f} smooth={bh_smooth:.0f} ref={bh_ref:.0f}",
            (8,24),cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,200,200),1)
        cv2.putText(vis,
            f"dist_err={d_dist:+.0f}px conf={conf:.2f}",
            (8,48),cv2.FONT_HERSHEY_SIMPLEX,0.6,c,2)
        cv2.putText(vis,f"step={step}",
            (8,72),cv2.FONT_HERSHEY_SIMPLEX,0.45,(200,200,200),1)
    else:
        cv2.putText(vis,"NO BOX",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)
    return vis

def send_chassis(pnc, vx, dt):
    twist = agibot_gdk.Twist()
    twist.linear  = agibot_gdk.Vector3()
    twist.angular = agibot_gdk.Vector3()
    twist.linear.x = vx
    pnc.move_chassis(twist)
    time.sleep(dt)

def stop_chassis(pnc):
    twist = agibot_gdk.Twist()
    twist.linear  = agibot_gdk.Vector3()
    twist.angular = agibot_gdk.Vector3()
    pnc.move_chassis(twist)

def run_calibrate(camera, model):
    print("\n[标定] 把机器人推到理想距离，按回车拍图...")
    input()
    for _ in range(8):
        camera.get_latest_image(HEAD_CAM, 1000.0)
        time.sleep(0.12)

    # 连续拍5帧取平均bh，更稳定
    bh_list = []
    last_det = None
    for _ in range(20):
        bgr = grab_frame(camera)
        if bgr is None:
            continue
        det = detect_box(model, bgr)
        if det:
            _,_,_,bh,_ = det
            bh_list.append(bh)
            last_det = det
        time.sleep(0.15)
        if len(bh_list) >= 5:
            break

    if not bh_list:
        print("❌ 未检测到箱子")
        return False

    cx,cy,bw,_,conf = last_det
    bh_ref = round(float(np.mean(bh_list)), 1)
    h_img, w_img = bgr.shape[:2]

    calib = {
        "cx":    round(cx,1),
        "cy":    round(cy,1),
        "bw":    round(bw,1),
        "bh":    bh_ref,
        "angle": 0.0,
        "img_w": w_img,
        "img_h": h_img,
    }
    with open(CALIB_FILE,"w") as f:
        json.dump(calib,f,indent=2)

    print(f"\n✅ 标定完成！bh_ref={bh_ref:.1f}px（{len(bh_list)}帧平均）")
    print(f"   保存: {CALIB_FILE}")

    # 验证图
    bgr2 = grab_frame(camera)
    if bgr2 is not None:
        det2 = detect_box(model, bgr2)
        if det2:
            cx2,cy2,bw2,bh2,_ = det2
            vis = bgr2.copy()
            x1=int(cx2-bw2/2); y1=int(cy2-bh2/2)
            x2=int(cx2+bw2/2); y2=int(cy2+bh2/2)
            cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),3)
            cv2.putText(vis,f"bh_ref={bh_ref:.0f}px",
                        (8,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cv2.imwrite("calib_result.jpg", vis)
            print("   验证图: calib_result.jpg")
    return True

def run_align(camera, model, pnc, calib, args):
    bh_ref = calib["bh"]
    print(f"\n[对齐] 标定bh_ref={bh_ref:.0f}px，容忍±{TOL_DIST}px")
    print(f"       bh越大=越近，bh越小=越远\n")

    if args.debug:
        os.makedirs("align_debug", exist_ok=True)

    no_box_cnt = 0
    aligned = False
    bh_buf = []  # 滑动窗口平均，稳定bh

    for step in range(MAX_STEPS):
        bgr = grab_frame(camera)
        if bgr is None:
            time.sleep(0.2)
            continue

        det = detect_box(model, bgr)

        if det is None:
            no_box_cnt += 1
            print(f"  [{step:03d}] ❌ 未检测到 ({no_box_cnt}/{NO_BOX_MAX})")
            if no_box_cnt >= NO_BOX_MAX:
                print("  ⚠️  多帧未检测到，停止")
                break
            time.sleep(0.1)
            if args.debug:
                vis = draw_debug(bgr, None, calib, step,
                                 bh_buf[-1] if bh_buf else bh_ref)
                cv2.imwrite(f"align_debug/step_{step:03d}.jpg", vis)
            continue

        no_box_cnt = 0
        cx,cy,bw,bh,conf = det

        # 滑动窗口平均（3帧），稳定bh
        bh_buf.append(bh)
        if len(bh_buf) > 3:
            bh_buf.pop(0)
        bh_smooth = float(np.mean(bh_buf))

        d_dist = bh_smooth - bh_ref   # 正=太近→后退，负=太远→前进

        ok_dist = abs(d_dist) <= TOL_DIST

        print(f"  [{step:03d}] "
              f"bh={bh:.0f} smooth={bh_smooth:.0f} "
              f"err={d_dist:+.0f}({'✅' if ok_dist else '❌'}) "
              f"conf={conf:.2f}")

        if args.debug:
            vis = draw_debug(bgr, det, calib, step, bh_smooth)
            cv2.imwrite(f"align_debug/step_{step:03d}.jpg", vis)

        if ok_dist:
            print(f"\n  ✅ 距离对齐完成！bh_smooth={bh_smooth:.0f} ref={bh_ref:.0f}")
            aligned = True
            break

        if args.no_move:
            time.sleep(0.1)
            continue

        # 前后控制
        # d_dist>0 太近 → 后退 vx<0
        # d_dist<0 太远 → 前进 vx>0
        speed = VX_BASE * (1.5 if abs(d_dist) > 30 else 1.0)
        import math
        vx = -math.copysign(speed, d_dist)
        send_chassis(pnc, vx, DT)

    else:
        print(f"\n  ⚠️  达到最大步数")

    stop_chassis(pnc)
    time.sleep(0.3)

    # 最终确认
    print("\n[最终确认]...")
    for _ in range(5):
        camera.get_latest_image(HEAD_CAM, 1000.0)
        time.sleep(0.1)
    bgr = grab_frame(camera)
    if bgr is not None:
        det = detect_box(model, bgr)
        if det:
            _,_,_,bh,conf = det
            print(f"  最终bh={bh:.0f}px  ref={bh_ref:.0f}px  "
                  f"err={bh-bh_ref:+.0f}px  conf={conf:.2f}")
            vis = draw_debug(bgr, det, calib, -1, bh)
            cv2.imwrite("align_result.jpg", vis)
            print("  结果图: align_result.jpg")
    return aligned

def main():
    args = parse_args()
    if not os.path.exists(args.model):
        print(f"❌ 模型不存在: {args.model}")
        sys.exit(1)
    from ultralytics import YOLO
    print("[YOLO] 加载模型...")
    model = YOLO(args.model)
    print("[YOLO] ✅ 就绪")

    agibot_gdk.gdk_init()
    robot  = agibot_gdk.Robot()
    camera = agibot_gdk.Camera()
    pnc    = agibot_gdk.Pnc()
    time.sleep(2)

    print("[头部] 低头 0.5 rad...")
    robot.move_head_joint(HEAD_LOOK_DOWN, HEAD_SPEED)
    time.sleep(2)

    print("[相机] 预热...")
    for _ in range(10):
        camera.get_latest_image(HEAD_CAM, 1000.0)
        time.sleep(0.15)
    print("[相机] ✅")

    if args.calibrate:
        run_calibrate(camera, model)
    else:
        if not os.path.exists(CALIB_FILE):
            print("❌ 找不到标定文件，请先运行:")
            print("   python3 box_align.py --calibrate")
            sys.exit(1)
        with open(CALIB_FILE) as f:
            calib = json.load(f)
        print(f"[标定] bh_ref={calib['bh']}px")

        if not args.no_move:
            try:
                task = pnc.get_task_state()
                if task.state in (1,2):
                    pnc.cancel_task(task.id)
                    time.sleep(0.5)
            except Exception:
                pass
            pnc.request_chassis_control(1)
            time.sleep(0.5)
            print("[底盘] ✅ 蟹行控制权获取")

        aligned = run_align(camera, model, pnc, calib, args)

        if not args.no_move:
            stop_chassis(pnc)
            try:
                pnc.cancel_task(pnc.get_task_state().id)
            except Exception:
                pass

        print(f"\n{'✅ 对齐成功' if aligned else '⚠️  对齐未完成'}")

    try:
        camera.close_camera()
    except Exception:
        pass
    agibot_gdk.gdk_release()
    return 0

if __name__ == "__main__":
    sys.exit(main())
