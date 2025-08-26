#!/usr/bin/env python3
"""
建图程序 - 先重定位（可选），再开始建图
"""

import time
import math
import threading
import agibot_gdk

def get_yaw_from_quaternion(orientation):
    qx, qy, qz, qw = orientation.x, orientation.y, orientation.z, orientation.w
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

def pose_printer(slam, stop_event):
    print("实时位姿（按回车停止建图）：")
    while not stop_event.is_set():
        try:
            odom = slam.get_odom_info()
            if odom is None or odom.pose is None:
                print("\r等待 SLAM 数据...", end="", flush=True)
                time.sleep(0.5)
                continue
            x = odom.pose.pose.position.x
            y = odom.pose.pose.position.y
            yaw = get_yaw_from_quaternion(odom.pose.pose.orientation)
            print(f"\rx={x:.3f} m, y={y:.3f} m, yaw={yaw:.3f} rad", end="", flush=True)
        except Exception as e:
            print(f"\r位姿获取失败: {e}，等待重试...", end="", flush=True)
        time.sleep(0.2)
    print()

def wait_for_relocalization(slam, map_mgr, target_map_id=None, timeout=60):
    # 如果有指定地图，先切换
    if target_map_id is not None:
        try:
            print(f"切换到地图 ID {target_map_id}...")
            map_mgr.switch_map(target_map_id)
            time.sleep(1)
        except Exception as e:
            print(f"切换地图失败: {e}")
            return False
    else:
        # 获取当前地图
        try:
            curr = map_mgr.get_curr_map()
            print(f"当前地图 ID={curr.id}, 名称={curr.name}")
        except:
            print("无法获取当前地图")
            return False

    print("等待重定位成功（loc_state=1）...")
    print("如果长时间无响应，请缓慢移动机器人或旋转，帮助匹配地图")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            odom = slam.get_odom_info()
            if odom is not None and odom.pose is not None:
                if odom.loc_state == 1:
                    print("\n重定位成功！")
                    return True
                else:
                    print(f"\r定位状态: {odom.loc_state}，等待中...", end="")
            else:
                print("\r里程计为空，等待数据...", end="")
        except Exception as e:
            print(f"\r获取里程计异常: {e}", end="")
        time.sleep(0.5)
    print("\n重定位超时")
    return False

def main():
    # 初始化 GDK
    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        print("GDK 初始化失败")
        return

    slam = agibot_gdk.Slam()
    pnc = agibot_gdk.Pnc()
    map_mgr = agibot_gdk.Map()
    time.sleep(2)

    # 1. 清除可能的残留任务
    try:
        task_state = pnc.get_task_state()
        if task_state and task_state.id != 0:
            print(f"发现未完成任务 ID={task_state.id}，正在取消...")
            pnc.cancel_task(task_state.id)
            time.sleep(1)
    except Exception as e:
        print(f"清理任务失败: {e}")

    # 2. 确保没有在建图
    try:
        slam.stop_mapping()
        time.sleep(0.5)
    except:
        pass

    # 3. 请求底盘控制（建图时需要移动）
    try:
        pnc.request_chassis_control(0)
        time.sleep(0.5)
    except Exception as e:
        print(f"请求控制失败: {e}")

    # 4. 获取所有地图，让用户选择重定位的地图（如果没有地图则直接建图）
    all_maps = []
    try:
        all_maps = map_mgr.get_all_map()
        print("可用地图列表:")
        for i, m in enumerate(all_maps):
            print(f"  {i+1}. ID={m.id}, 名称='{m.name}'")
    except Exception as e:
        print(f"获取地图列表失败: {e}")

    # 5. 重定位（如果有地图）
    reloc_success = False
    if all_maps:
        # 默认选择第一个地图
        target = all_maps[3]
        print(f"\n将使用地图 ID {target.id} 进行重定位")
        reloc_success = wait_for_relocalization(slam, map_mgr, target.id, timeout=60)
        if not reloc_success:
            print("重定位失败，是否继续建图？(y/n)")
            choice = input().strip().lower()
            if choice != 'y':
                agibot_gdk.gdk_release()
                return
        else:
            print("\n重定位成功，接下来将开始建图。")

            input("按回车键继续...")
    else:
        print("未找到任何地图，将直接开始建图（无重定位）")

    # 6. 开始建图
    print("开始建图...")
    try:
        slam.start_mapping()
    except Exception as e:
        print(f"开始建图失败: {e}")
        agibot_gdk.gdk_release()
        return

    # 7. 启动位姿打印线程
    stop_print = threading.Event()
    printer_thread = threading.Thread(target=pose_printer, args=(slam, stop_print), daemon=True)
    printer_thread.start()

    print("请移动机器人覆盖需要建图的区域...")
    try:
        input("完成后按回车键停止建图")
    except KeyboardInterrupt:
        print("\n用户中断建图")

    # 停止打印
    stop_print.set()
    printer_thread.join(timeout=1)

    # 停止建图
    print("\n停止建图...")
    try:
        slam.stop_mapping()
    except Exception as e:
        print(f"停止建图失败: {e}")

    time.sleep(1)

    # 显示保存的地图
    try:
        all_maps = map_mgr.get_all_map()
        if all_maps:
            print("\n已保存的地图列表:")
            for m in all_maps:
                print(f"  ID={m.id}, 名称={m.name}, 当前={m.is_curr_map}")
            curr_map = map_mgr.get_curr_map()
            print(f"当前地图 ID={curr_map.id}, 名称={curr_map.name}")
        else:
            print("未找到保存的地图，请检查建图流程")
    except Exception as e:
        print(f"获取地图列表失败: {e}")

    # 停止运动
    try:
        twist = agibot_gdk.Twist()
        twist.linear.x = 0.0
        pnc.move_chassis(twist)
    except:
        pass

    agibot_gdk.gdk_release()
    print("建图结束")

if __name__ == "__main__":
    main()