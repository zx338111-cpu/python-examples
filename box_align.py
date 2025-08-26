#!/usr/bin/env python3
import argparse, json, math, os, time, sys
import cv2, numpy as np
import agibot_gdk

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
CALIB_FILE = os.path.join(os.path.dirname(__file__), "calib.json")
HEAD_CAM       = agibot_gdk.CameraType.kHeadColor
HEAD_LOOK_DOWN = [0.0, 0.0, 0.5]
HEAD_SPEED     = [0.2, 0.2, 0.2]
CONF_THR  = 0.35
TOL_DX    = 8
TOL_DIST  = 10
VY_BASE   = 0.06
VX_BASE   = 0.05
DT        = 0.12
MAX_STEPS   = 150
NO_BOX_MAX  = 20

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
    """只检测位置和大小，不算角度（角度不稳定先去掉）"""
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

def draw_debug(bgr, det, calib, step):
    vis = bgr.copy()
    h, w = vis.shape[:2]
    cx_ref = calib["cx"]
    bh_ref = calib["bh"]
    cv2.line(vis,(int(cx_ref),0),(int(cx_ref),h),(255,100,0),2)
    if det:
        cx,cy,bw,bh,conf = det
        x1=int(cx-bw/2); y1=int(cy-bh/2)
        x2=int(cx+bw/2); y2=int(cy+bh/2)
        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.circle(vis,(int(cx),int(cy)),6,(0,0,255),-1)
        cv2.line(vis,(int(cx_ref),int(cy)),(int(cx),int(cy)),(0,255,255),2)
        dx     = cx - cx_ref
        d_dist = bh - bh_ref
        c_dx = (0,255,0) if abs(dx)<TOL_DX else (0,0,255)
        c_d  = (0,255,0) if abs(d_dist)<TOL_DIST else (0,0,255)
        cv2.putText(vis,f"dx={dx:+.0f}px",(8,24),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,c_dx,2)
        cv2.putText(vis,f"dist_err={d_dist:+.0f}px bh={bh:.0f}(ref={bh_ref:.0f})",
                    (8,50),cv2.FONT_HERSHEY_SIMPLEX,0.6,c_d,2)
        cv2.putText(vis,f"conf={conf:.2f} step={step}",
                    (8,76),cv2.FONT_HERSHEY_SIMPLEX,0.45,(200,200,200),1)
    else:
        cv2.putText(vis,"NO BOX",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)
    return vis

def send_chassis(pnc, vx, vy, dt):
    twist = agibot_gdk.Twist()
    twist.linear  = agibot_gdk.Vector3()
    twist.angular = agibot_gdk.Vector3()
    twist.linear.x = vx
    twist.linear.y = vy
    pnc.move_chassis(twist)
    time.sleep(dt)

def stop_chassis(pnc):
    twist = agibot_gdk.Twist()
    twist.linear  = agibot_gdk.Vector3()
    twist.angular = agibot_gdk.Vector3()
    pnc.move_chassis(twist)

def run_calibrate(camera, model):
    print("\n[标定] 确认机器人已推到理想夹取位置")
    print("[标定] 按回车拍图标定...")
    input()
    for _ in range(8):
        camera.get_latest_image(HEAD_CAM, 1000.0)
        time.sleep(0.12)
    bgr = grab_frame(camera)
    if bgr is None:
        print("❌ 拍图失败"); return False
    det = detect_box(model, bgr)
    if det is None:
        print("❌ 未检测到箱子")
        cv2.imwrite("calib_failed.jpg", bgr); return False
    cx,cy,bw,bh,conf = det
    h,w = bgr.shape[:2]
    calib = {"cx":round(cx,1),"cy":round(cy,1),
             "bw":round(bw,1),"bh":round(bh,1),
             "angle":0.0,"img_w":w,"img_h":h}
    with open(CALIB_FILE,"w") as f:
        json.dump(calib,f,indent=2)
    print(f"\n✅ 标定完成！cx={cx:.0f} cy={cy:.0f} bw={bw:.0f} bh={bh:.0f}")
    print(f"   保存: {CALIB_FILE}")
    vis = bgr.copy()
    x1=int(cx-bw/2); y1=int(cy-bh/2)
    x2=int(cx+bw/2); y2=int(cy+bh/2)
    cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),3)
    cv2.circle(vis,(int(cx),int(cy)),8,(0,0,255),-1)
    cv2.line(vis,(int(cx),0),(int(cx),h),(255,100,0),2)
    cv2.putText(vis,f"CALIB cx={cx:.0f} bh={bh:.0f}",
                (8,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    cv2.imwrite("calib_result.jpg", vis)
    print("   验证图: calib_result.jpg")
    return True

def run_align(camera, model, pnc, calib, args):
    cx_ref = calib["cx"]
    bh_ref = calib["bh"]
    print(f"\n[对齐] 标定参考: cx={cx_ref:.0f}px  bh={bh_ref:.0f}px")
    print(f"[对齐] 容忍: 左右±{TOL_DX}px  距离±{TOL_DIST}px\n")
    if args.debug:
        os.makedirs("align_debug", exist_ok=True)
    no_box_cnt = 0
    aligned = False

    for step in range(MAX_STEPS):
        bgr = grab_frame(camera)
        if bgr is None:
            time.sleep(0.2); continue

        det = detect_box(model, bgr)

        if args.debug:
            vis = draw_debug(bgr, det, calib, step)
            cv2.imwrite(f"align_debug/step_{step:03d}.jpg", vis)

        if det is None:
            no_box_cnt += 1
            print(f"  [{step:03d}] ❌ 未检测到 ({no_box_cnt}/{NO_BOX_MAX})")
            if no_box_cnt >= NO_BOX_MAX:
                print("  ⚠️  多帧未检测到，停止"); break
            time.sleep(0.15); continue

        no_box_cnt = 0
        cx,cy,bw,bh,conf = det
        dx     = cx - cx_ref    # 正=箱子偏右→底盘右移(vy<0)
        d_dist = bh - bh_ref    # 正=箱子变大=太近→后退(vx<0)

        ok_dx   = abs(dx)     <= TOL_DX
        ok_dist = abs(d_dist) <= TOL_DIST

        print(f"  [{step:03d}] "
              f"dx={dx:+.0f}({'✅' if ok_dx else '❌'}) "
              f"dist={d_dist:+.0f}bh={bh:.0f}({'✅' if ok_dist else '❌'}) "
              f"conf={conf:.2f}")

        if ok_dx and ok_dist:
            print(f"\n  ✅ 对齐完成！dx={dx:+.0f}px  dist={d_dist:+.0f}px")
            aligned = True; break

        if args.no_move:
            time.sleep(0.1); continue

        vx, vy = 0.0, 0.0

        # 左右平移：dx>0箱子偏右→底盘右移(vy负)
        if not ok_dx:
            vy = -math.copysign(
                VY_BASE * (1.5 if abs(dx) > 50 else 1.0), dx)

        # 前后：d_dist>0太近→后退(vx负)，d_dist<0太远→前进(vx正)
        if not ok_dist:
            vx = -math.copysign(VX_BASE, d_dist)

        send_chassis(pnc, vx, vy, DT)

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
            cx,cy,bw,bh,conf = det
            dx     = cx - cx_ref
            d_dist = bh - bh_ref
            print(f"  最终: dx={dx:+.0f}px  dist={d_dist:+.0f}px  bh={bh:.0f}px")
            vis = draw_debug(bgr, det, calib, -1)
            cv2.imwrite("align_result.jpg", vis)
            print("  结果图: align_result.jpg")
    return aligned

def main():
    args = parse_args()
    if not os.path.exists(args.model):
        print(f"❌ 模型不存在: {args.model}"); sys.exit(1)
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
            print(f"❌ 找不到标定文件，请先运行:")
            print(f"   python3 box_align.py --calibrate")
            sys.exit(1)
        with open(CALIB_FILE) as f:
            calib = json.load(f)
        print(f"[标定] cx={calib['cx']}  bh={calib['bh']}")
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
