#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, sys, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import cv2
    from ultralytics import YOLO
    import agibot_gdk
    from g2_robot_controller import G2RobotController
except Exception as e:
    print(f"❌ 依赖加载失败: {e}")
    sys.exit(1)

class BoxVisualAlignAndGraspV4:
    def __init__(self, model_path:str, class_name:str='box', camera_name:str='right',
                 conf:float=0.35, retries:int=20, timeout_ms:float=1000.0,
                 save_dir:str='align_debug_yolo', align_loops:int=2,
                 stop_ex_px:int=12, stop_ey_px:int=12, stop_eh_px:int=10,
                 low_conf_stop:float=0.70, max_step_dx:float=0.008,
                 max_step_dy:float=0.010, max_step_dz:float=0.006,
                 do_grasp:bool=False, close_position:float=1.0,
                 right_bias_dx:float=0.0, right_bias_dy:float=0.0, right_bias_dz:float=0.0):
        self.model_path = model_path
        self.class_name = class_name
        self.camera_name = camera_name.lower().strip()
        self.conf = conf
        self.retries = retries
        self.timeout_ms = timeout_ms
        self.save_dir = Path(save_dir); self.save_dir.mkdir(parents=True, exist_ok=True)
        self.align_loops = align_loops
        self.stop_ex_px = stop_ex_px
        self.stop_ey_px = stop_ey_px
        self.stop_eh_px = stop_eh_px
        self.low_conf_stop = low_conf_stop
        self.max_step_dx = max_step_dx
        self.max_step_dy = max_step_dy
        self.max_step_dz = max_step_dz
        self.do_grasp = do_grasp
        self.close_position = close_position
        self.TARGET_CX = 640.0
        self.TARGET_CY = 410.0
        self.TARGET_H = 208.0
        self.RIGHT_ARM_BIAS_DX = right_bias_dx
        self.RIGHT_ARM_BIAS_DY = right_bias_dy
        self.RIGHT_ARM_BIAS_DZ = right_bias_dz
        self.ctrl: Optional[G2RobotController] = None
        self.camera = None
        self.model = None
        self.camera_type = self._resolve_camera_type(self.camera_name)

    def _resolve_camera_type(self, name:str):
        if name == 'left':
            return agibot_gdk.CameraType.kHandLeftColor
        if name == 'right':
            return agibot_gdk.CameraType.kHandRightColor
        raise ValueError('camera 只能是 left 或 right')

    def init(self):
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f'找不到模型文件: {self.model_path}')
        print('🔧 初始化控制器...')
        self.ctrl = G2RobotController()
        self.ctrl.init()
        print('📷 初始化相机...')
        self.camera = agibot_gdk.Camera()
        time.sleep(2.0)
        self.model = YOLO(str(self.model_path))
        print(f'🧠 YOLO 模型已加载: {self.model_path}')
        try:
            intr = self.camera.get_camera_intrinsic(self.camera_type)
            fx, fy, cx, cy = list(intr.intrinsic)[:4]
            print(f'📐 相机内参: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}')
        except Exception as e:
            print(f'⚠️ 获取相机内参失败: {e}')

    def release(self):
        if self.ctrl is not None:
            self.ctrl.release()

    def decode_image(self, image):
        if image is None or not hasattr(image, 'data') or image.data is None or len(image.data) == 0:
            return None
        try:
            if image.encoding == agibot_gdk.Encoding.JPEG:
                return cv2.imdecode(np.frombuffer(image.data, np.uint8), cv2.IMREAD_COLOR)
            if image.encoding == agibot_gdk.Encoding.PNG:
                return cv2.imdecode(np.frombuffer(image.data, np.uint8), cv2.IMREAD_COLOR)
            if image.encoding == agibot_gdk.Encoding.UNCOMPRESSED:
                if image.color_format == agibot_gdk.ColorFormat.RGB:
                    arr = np.frombuffer(image.data, dtype=np.uint8).reshape((image.height, image.width, 3))
                    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                if image.color_format == agibot_gdk.ColorFormat.BGR:
                    return np.frombuffer(image.data, dtype=np.uint8).reshape((image.height, image.width, 3))
                if image.color_format == agibot_gdk.ColorFormat.GRAY8:
                    arr = np.frombuffer(image.data, dtype=np.uint8).reshape((image.height, image.width))
                    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            print(f'⚠️ 图像解码失败: {e}')
        return None

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], Optional[Any]]:
        try:
            image = self.camera.get_latest_image(self.camera_type, self.timeout_ms)
        except Exception as e:
            print(f'⚠️ 取图失败: {e}')
            return None, None
        return self.decode_image(image), image

    def warmup_camera(self, n:int=20, dt:float=0.05):
        for _ in range(n):
            self.get_latest_frame()
            time.sleep(dt)

    def goto_view_pose(self):
        self.ctrl.move_waist_to_point('view_waist')
        time.sleep(1.2)
        self.ctrl.move_both_arms_ee_to_point('view_pose_calib', duration_s=2.0)
        time.sleep(2.0)
        self.warmup_camera()

    def goto_grasp_pose(self):
        self.ctrl.move_waist_to_point('grasp_waist')
        time.sleep(0.8)
        self.ctrl.move_both_arms_ee_to_point('grasp_pose', duration_s=2.0)
        time.sleep(1.0)

    def detect_once(self, frame:np.ndarray):
        results = self.model(frame, conf=self.conf, verbose=False)
        best = None
        best_conf = -1.0
        for r in results or []:
            names = r.names
            for box in r.boxes:
                cls_id = int(box.cls[0]); cls_name = names[cls_id]; conf = float(box.conf[0])
                if cls_name != self.class_name:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                cx = int((x1 + x2) / 2); cy = int((y1 + y2) / 2)
                w = int(x2 - x1); h = int(y2 - y1)
                if conf > best_conf:
                    best_conf = conf
                    best = {'class_name': cls_name, 'conf': conf, 'bbox_xyxy': [x1, y1, x2, y2],
                            'center_xy': [cx, cy], 'size_wh': [w, h]}
        return best

    def detect_box_stable(self, allow_reseat:bool=True):
        dets: List[Dict[str, Any]] = []
        print(f'🎯 开始稳定检测，camera={self.camera_name}, retries={self.retries}')
        for i in range(self.retries):
            frame, image_obj = self.get_latest_frame()
            if frame is None:
                print(f'  [{i+1}/{self.retries}] 未取到有效图像'); time.sleep(0.15); continue
            det = self.detect_once(frame)
            if det is None:
                print(f'  [{i+1}/{self.retries}] 未检测到目标类: {self.class_name}'); time.sleep(0.15); continue
            print(f"  [{i+1}/{self.retries}] conf={det['conf']:.3f}, center={det['center_xy']}, size={det['size_wh']}")
            det['_frame'] = frame.copy(); det['_image_obj'] = image_obj
            dets.append(det); time.sleep(0.05)
        if not dets and allow_reseat:
            print('⚠️ 第一次到位后未检出，执行二次到位 + 相机预热，再试一次')
            self.ctrl.move_both_arms_ee_to_point('view_pose_calib', duration_s=1.5)
            time.sleep(2.0); self.warmup_camera()
            return self.detect_box_stable(False)
        if not dets:
            raise RuntimeError('多次尝试后仍未检测到目标')
        cxs = np.array([d['center_xy'][0] for d in dets], dtype=float)
        cys = np.array([d['center_xy'][1] for d in dets], dtype=float)
        ws = np.array([d['size_wh'][0] for d in dets], dtype=float)
        hs = np.array([d['size_wh'][1] for d in dets], dtype=float)
        confs = np.array([d['conf'] for d in dets], dtype=float)
        cx = int(round(np.median(cxs))); cy = int(round(np.median(cys)))
        w = int(round(np.median(ws))); h = int(round(np.median(hs)))
        conf_med = float(np.median(confs))
        best_idx = int(np.argmin((cxs - cx) ** 2 + (cys - cy) ** 2))
        chosen = dets[best_idx]
        return {'class_name': self.class_name, 'conf': conf_med, 'bbox_xyxy': chosen['bbox_xyxy'],
                'center_xy': [cx, cy], 'size_wh': [w, h], '_frame': chosen['_frame'], '_image_obj': chosen['_image_obj']}

    def compute_step_delta(self, cx:float, cy:float, h:float):
        ex = cx - self.TARGET_CX
        ey = cy - self.TARGET_CY
        eh = h - self.TARGET_H
        dx = np.clip(0.0008 * ex, -self.max_step_dx, self.max_step_dx)
        dy = np.clip(-0.00037 * ey, -self.max_step_dy, self.max_step_dy)
        dz = np.clip(0.0002 * eh, -self.max_step_dz, self.max_step_dz)
        return float(dx), float(dy), float(dz)

    def draw_detection(self, frame:np.ndarray, det:Dict[str, Any], total_dx:float, total_dy:float, total_dz:float):
        img = frame.copy()
        x1, y1, x2, y2 = det['bbox_xyxy']
        cx, cy = det['center_xy']; _, h = det['size_wh']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.circle(img, (cx, cy), 5, (0,0,255), -1)
        cv2.circle(img, (int(self.TARGET_CX), int(self.TARGET_CY)), 5, (255,255,0), -1)
        cv2.putText(img, f'curr=({cx},{cy},{h}) target=({int(self.TARGET_CX)},{int(self.TARGET_CY)},{int(self.TARGET_H)})',
                    (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
        cv2.putText(img, f'view total: dx={total_dx:+.4f} dy={total_dy:+.4f} dz={total_dz:+.4f}',
                    (20,65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,0,255), 2)
        cv2.putText(img, f'right bias: dx={self.RIGHT_ARM_BIAS_DX:+.4f} dy={self.RIGHT_ARM_BIAS_DY:+.4f} dz={self.RIGHT_ARM_BIAS_DZ:+.4f}',
                    (20,95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
        return img

    def save_debug_image(self, image_obj, img:np.ndarray, prefix:str):
        ts = getattr(image_obj, 'timestamp_ns', None) or int(time.time() * 1e9)
        out = self.save_dir / f'{prefix}_{self.camera_name}_{ts}.jpg'
        cv2.imwrite(str(out), img)
        return out

    def run(self):
        print('\n========== 第 1 步：到拍照位 ==========')
        self.goto_view_pose()
        print('\n========== 第 2 步：拍照位迭代修正 ==========')
        total_dx = total_dy = total_dz = 0.0
        final_det = None
        prev_good_det = None
        for i in range(self.align_loops):
            print(f'\n------ 迭代 {i+1}/{self.align_loops} ------')
            det = self.detect_box_stable(); final_det = det
            cx, cy = det['center_xy']; _, h = det['size_wh']; conf = det['conf']
            ex = cx - self.TARGET_CX; ey = cy - self.TARGET_CY; eh = h - self.TARGET_H
            print(f'当前误差: ex={ex:+.1f}px, ey={ey:+.1f}px, eh={eh:+.1f}px, conf={conf:.3f}')
            if conf >= self.low_conf_stop:
                prev_good_det = det
            if abs(ex) < self.stop_ex_px and abs(ey) < self.stop_ey_px and abs(eh) < self.stop_eh_px:
                print('✅ 拍照位误差已足够小，停止迭代')
                break
            if conf < self.low_conf_stop:
                print('⚠️ 置信度过低，停止继续迭代，避免把相机带出有效视野')
                break
            step_dx, step_dy, step_dz = self.compute_step_delta(cx, cy, h)
            print(f'本轮修正: dx={step_dx:+.4f} m, dy={step_dy:+.4f} m, dz={step_dz:+.4f} m')
            self.ctrl.move_both_arms_delta(dx=step_dx, dy=step_dy, dz=step_dz, duration_s=1.0)
            time.sleep(1.2)
            total_dx += step_dx; total_dy += step_dy; total_dz += step_dz
        if prev_good_det is not None:
            final_det = prev_good_det
        if final_det is None:
            raise RuntimeError('拍照位迭代中没有得到任何有效检测结果')
        print('\n========== 第 3 步：记录累计修正 ==========')
        print(f'累计修正: total_dx={total_dx:+.4f}, total_dy={total_dy:+.4f}, total_dz={total_dz:+.4f}')
        dbg = self.draw_detection(final_det['_frame'], final_det, total_dx, total_dy, total_dz)
        dbg_path = self.save_debug_image(final_det['_image_obj'], dbg, 'align_result')
        print(f'调试图已保存: {dbg_path}')
        print('\n========== 第 4 步：到抓取位并叠加累计修正 ==========')
        self.goto_grasp_pose()
        self.ctrl.move_both_arms_delta(dx=total_dx, dy=total_dy, dz=total_dz, duration_s=1.0)
        time.sleep(1.0)
        if abs(self.RIGHT_ARM_BIAS_DX) > 1e-9 or abs(self.RIGHT_ARM_BIAS_DY) > 1e-9 or abs(self.RIGHT_ARM_BIAS_DZ) > 1e-9:
            print('\n========== 第 4.5 步：右臂单独补偿 ==========')
            print(f'右臂 bias: dx={self.RIGHT_ARM_BIAS_DX:+.4f}, dy={self.RIGHT_ARM_BIAS_DY:+.4f}, dz={self.RIGHT_ARM_BIAS_DZ:+.4f}')
            self.ctrl.move_right_arm_delta(dx=self.RIGHT_ARM_BIAS_DX, dy=self.RIGHT_ARM_BIAS_DY,
                                           dz=self.RIGHT_ARM_BIAS_DZ, duration_s=0.8)
            time.sleep(0.8)
        if self.do_grasp:
            print('\n========== 第 5 步：闭夹爪 ==========')
            self.ctrl.close_gripper(position=self.close_position)
            time.sleep(1.0)
            print('✅ 已执行抓取')
        else:
            print('\n⚠️ 当前为 dry-run，只走到了修正后的抓取位，没有闭夹爪。')
            print('   确认位置无误后，加 --do-grasp 再执行真正抓取。')

def build_argparser():
    ap = argparse.ArgumentParser(description='拍照位迭代修正 -> 修正抓取位 -> 抓取（V4，手工三轴控制）')
    ap.add_argument('--model', required=True)
    ap.add_argument('--class-name', default='box')
    ap.add_argument('--camera', default='right', choices=['left','right'])
    ap.add_argument('--conf', type=float, default=0.35)
    ap.add_argument('--retries', type=int, default=20)
    ap.add_argument('--timeout-ms', type=float, default=1000.0)
    ap.add_argument('--save-dir', default='align_debug_yolo')
    ap.add_argument('--align-loops', type=int, default=2)
    ap.add_argument('--stop-ex-px', type=int, default=12)
    ap.add_argument('--stop-ey-px', type=int, default=12)
    ap.add_argument('--stop-eh-px', type=int, default=10)
    ap.add_argument('--low-conf-stop', type=float, default=0.70)
    ap.add_argument('--max-step-dx', type=float, default=0.008)
    ap.add_argument('--max-step-dy', type=float, default=0.010)
    ap.add_argument('--max-step-dz', type=float, default=0.006)
    ap.add_argument('--right-bias-dx', type=float, default=0.0)
    ap.add_argument('--right-bias-dy', type=float, default=0.0)
    ap.add_argument('--right-bias-dz', type=float, default=0.0)
    ap.add_argument('--do-grasp', action='store_true')
    ap.add_argument('--close-position', type=float, default=1.0)
    return ap

def main():
    args = build_argparser().parse_args()
    app = BoxVisualAlignAndGraspV4(
        model_path=args.model, class_name=args.class_name, camera_name=args.camera,
        conf=args.conf, retries=args.retries, timeout_ms=args.timeout_ms,
        save_dir=args.save_dir, align_loops=args.align_loops,
        stop_ex_px=args.stop_ex_px, stop_ey_px=args.stop_ey_px, stop_eh_px=args.stop_eh_px,
        low_conf_stop=args.low_conf_stop, max_step_dx=args.max_step_dx,
        max_step_dy=args.max_step_dy, max_step_dz=args.max_step_dz,
        do_grasp=args.do_grasp, close_position=args.close_position,
        right_bias_dx=args.right_bias_dx, right_bias_dy=args.right_bias_dy, right_bias_dz=args.right_bias_dz,
    )
    try:
        app.init(); app.run()
    except KeyboardInterrupt:
        print('\n⚠️ 用户中断')
    except Exception as e:
        print(f'\n❌ 运行失败: {e}')
        raise
    finally:
        app.release()

if __name__ == '__main__':
    main()