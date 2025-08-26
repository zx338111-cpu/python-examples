#!/usr/bin/env python3
"""
导航到标定点程序
使用方法: python navigate_to_calibration_point.py --x 1.0 --y 2.0 --yaw 0.785
参数说明:
  --x        目标点 x 坐标 (米)
  --y        目标点 y 坐标 (米)
  --yaw      目标点朝向 (弧度，偏航角)
  --map_id   地图 ID (可选，默认使用当前地图)
  --timeout  导航超时时间 (秒，默认 60)
"""

import time
import argparse
import math
import agibot_gdk

class Navigator:
    def __init__(self):
        if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
            raise RuntimeError("GDK 初始化失败")
        self.slam = agibot_gdk.Slam()
        self.pnc = agibot_gdk.Pnc()
        self.map = agibot_gdk.Map()
        time.sleep(2)

    def switch_map(self, map_id):
        try:
            self.map.switch_map(map_id)
            print(f"已切换到地图 ID {map_id}")
            return True
        except Exception as e:
            print(f"地图切换失败: {e}")
            return False

    def request_control(self, control_mode=0):
        """请求底盘控制权限"""
        print(f"请求底盘控制权限，模式 {control_mode}...")
        self.pnc.request_chassis_control(control_mode)
        time.sleep(0.5)

    def wait_for_localization(self, timeout=10):
        """等待 SLAM 定位成功"""
        print("等待 SLAM 定位稳定...")
        start = time.time()
        while time.time() - start < timeout:
            state = self.slam.get_slam_state()
            odom = self.slam.get_odom_info()
            # 假设 loc_state 为 1 表示定位成功
            if odom.loc_state == 1:
                print("定位成功")
                return True
            time.sleep(0.5)
        print("定位超时")
        return False

    def get_current_pose(self):
        """获取当前机器人位姿 (x, y, yaw)"""
        odom = self.slam.get_odom_info()
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y
        # 从四元数计算偏航角 (yaw)
        qx = odom.pose.pose.orientation.x
        qy = odom.pose.pose.orientation.y
        qz = odom.pose.pose.orientation.z
        qw = odom.pose.pose.orientation.w
        # 欧拉角转换 (只取偏航)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return x, y, yaw

    def navigate_to_goal(self, target_x, target_y, target_yaw, timeout=60):
        """
        导航到目标点并调整朝向
        :param target_x:  目标 x 坐标 (米)
        :param target_y:  目标 y 坐标 (米)
        :param target_yaw: 目标朝向 (弧度)
        :param timeout:    超时时间 (秒)
        :return: True 成功，False 失败
        """
        # 创建导航请求
        navi_req = agibot_gdk.NaviReq()
        navi_req.target.position.x = target_x
        navi_req.target.position.y = target_y
        navi_req.target.position.z = 0.0

        # 将目标朝向（偏航角）转换为四元数
        # 这里假设机器人只在平面运动，绕 Z 轴旋转
        half_yaw = target_yaw / 2.0
        qz = math.sin(half_yaw)
        qw = math.cos(half_yaw)
        navi_req.target.orientation.x = 0.0
        navi_req.target.orientation.y = 0.0
        navi_req.target.orientation.z = qz
        navi_req.target.orientation.w = qw

        print(f"发送导航请求: 目标 ({target_x:.2f}, {target_y:.2f}, 偏航 {target_yaw:.2f} rad)")
        # 开始导航
        self.pnc.normal_navi(navi_req)

        # 监控任务状态
        start_time = time.time()
        task_id = None
        while time.time() - start_time < timeout:
            task_state = self.pnc.get_task_state()
            if task_state is None:
                print("无法获取任务状态")
                break

            # 记录任务 ID
            if task_id is None:
                task_id = task_state.id
                print(f"任务 ID: {task_id}, 状态: {task_state.state}")

            # 判断任务是否完成
            # 根据 agibot_gdk 文档，state 可能为 0 表示运行中，1 表示成功，2 表示失败等
            if task_state.state == 1:   # 假设 1 为成功
                print("导航任务成功完成")
                # 任务完成后，检查最终位姿是否达到目标（可选）
                cur_x, cur_y, cur_yaw = self.get_current_pose()
                print(f"当前位姿: ({cur_x:.2f}, {cur_y:.2f}, {cur_yaw:.2f})")
                return True
            elif task_state.state == 2: # 假设 2 为失败
                print(f"导航任务失败: {task_state.error_msg}")
                return False

            # 每 0.5 秒检查一次状态
            time.sleep(0.5)

        print("导航超时")
        # 超时后取消任务
        if task_id is not None:
            self.pnc.cancel_task(task_id)
        return False

    def release(self):
        """释放资源"""
        # 取消所有任务
        try:
            task_state = self.pnc.get_task_state()
            if task_state and task_state.id != 0:
                self.pnc.cancel_task(task_state.id)
        except:
            pass
        # 释放 GDK
        agibot_gdk.gdk_release()
        print("资源已释放")

def main():
    parser = argparse.ArgumentParser(description="导航到标定点")
    parser.add_argument("--x", type=float, required=True, help="目标 x 坐标 (米)")
    parser.add_argument("--y", type=float, required=True, help="目标 y 坐标 (米)")
    parser.add_argument("--yaw", type=float, required=True, help="目标朝向 (弧度)")
    parser.add_argument("--map_id", type=str, default=None, help="地图 ID (可选)")
    parser.add_argument("--timeout", type=int, default=60, help="导航超时时间 (秒)")
    args = parser.parse_args()

    navigator = None
    try:
        navigator = Navigator()

        # 可选：切换地图
        if args.map_id is not None:
            if not navigator.switch_map(int(args.map_id)):
                return

        # 请求底盘控制
        navigator.request_control(control_mode=0)  # 0: 阿克曼模式

        # 等待定位稳定
        if not navigator.wait_for_localization(timeout=30):
            print("定位未成功，导航可能失败")

        # 开始导航
        success = navigator.navigate_to_goal(args.x, args.y, args.yaw, args.timeout)
        if success:
            print("成功到达标定点")
        else:
            print("导航失败")

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"程序出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if navigator:
            navigator.release()

if __name__ == "__main__":
    main()