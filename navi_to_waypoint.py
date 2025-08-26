#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
navi_to_waypoint.py  —  G2机器人：导航到地图中预设的导航点

使用前提：
  1. 已在G02 Pad上建好地图
  2. 已在Pad上完成重定位（这一步必须在Pad上做，代码无法替代）
  3. 已知道各导航点的地图坐标（x, y, z）和朝向（四元数 x,y,z,w）

如何获取导航点坐标？见文件末尾的 "获取坐标的方法"

用法：
  source ~/.cache/agibot/app/env.sh
  python3 navi_to_waypoint.py              # 交互式菜单
  python3 navi_to_waypoint.py --wp 0       # 直接导航到第0号导航点
  python3 navi_to_waypoint.py --wp 客厅     # 按名称导航
  python3 navi_to_waypoint.py --list       # 列出所有导航点
  python3 navi_to_waypoint.py --seq 0 1 2  # 依次导航到多个点
"""

import argparse
import time
import sys
from typing import Optional

import agibot_gdk


# ═══════════════════════════════════════════════════════════════
#  ★★★ 在这里填写你的导航点 ★★★
#
#  坐标来源：从Pad的地图界面读取，或用下方的"记录当前位置"功能获取
#
#  朝向说明（四元数）：
#    正北方向  → [0, 0, 0, 1]  (w=1, 其余0)
#    朝东90°  → [0, 0, 0.707, 0.707]
#    朝西90°  → [0, 0, -0.707, 0.707]
#    朝南180° → [0, 0, 1, 0]
#    如果不在意朝向，直接用 [0, 0, 0, 1] 即可
#
#  格式：
#  "名称": {
#      "position": [x, y, z],          # 单位：米
#      "orientation": [qx, qy, qz, qw] # 四元数，归一化
#  }
# ═══════════════════════════════════════════════════════════════

WAYPOINTS = {
    "起始点": {
        "position": [0.0, 0.0, 0.0],
        "orientation": [0.0, 0.0, 0.0, 1.0],
    },
    "A点": {
        "position": [2.0, 1.0, 0.0],
        "orientation": [0.0, 0.0, 0.0, 1.0],
    },
    "B点": {
        "position": [5.0, 3.0, 0.0],
        "orientation": [0.0, 0.0, 0.707, 0.707],
    },
    "充电桩": {
        "position": [-1.5, 0.5, 0.0],
        "orientation": [0.0, 0.0, 1.0, 0.0],
    },
    "1": {
        "position":    [0.185261, 0.406046, 0.003848],
        "orientation": [-0.000348, -4.3e-05, 0.013894, 0.999903],
    },
    "2": {
        "position":    [-0.758743, -6.222107, 0.014117],
        "orientation": [-0.004988, -0.001072, 0.897476, 0.441033],
    },
}

# ═══════════════════════════════════════════════════════════════
#  导航器类
# ═══════════════════════════════════════════════════════════════

class WaypointNavigator:
    # 任务状态码（来自GDK文档）
    STATE_IDLE     = 0
    STATE_STARTING = 1
    STATE_RUNNING  = 2
    STATE_PAUSING  = 3
    STATE_PAUSED   = 4
    STATE_RESUMING = 5
    STATE_CANCELING= 6
    STATE_CANCELED = 7
    STATE_FAILED   = 8
    STATE_SUCCESS  = 9

    STATE_NAMES = {
        0: "空闲", 1: "启动中", 2: "运行中", 3: "暂停中",
        4: "已暂停", 5: "恢复中", 6: "取消中", 7: "已取消",
        8: "失败", 9: "成功"
    }

    def __init__(self):
        print("正在初始化GDK...")
        agibot_gdk.gdk_init()
        self.pnc = agibot_gdk.Pnc()
        self.slam = agibot_gdk.Slam()
        self.map = agibot_gdk.Map()
        time.sleep(1.5)  # 等待DDS连接
        print("初始化完成")

    def show_map_info(self):
        """显示当前地图信息"""
        print("\n=== 地图信息 ===")
        try:
            all_maps = self.map.get_all_map()
            curr_map = self.map.get_curr_map()
            print(f"所有地图: {all_maps}")
            print(f"当前地图: {curr_map}")
        except Exception as e:
            print(f"获取地图信息失败: {e}")

    def show_current_pose(self):
        """显示机器人当前位置（可用于记录导航点坐标）"""
        print("\n=== 当前位置 ===")
        try:
            odom = self.slam.get_odom_info()
            pos = odom.pose.pose.position
            ori = odom.pose.pose.orientation
            print(f"位置:  x={pos.x:.4f}, y={pos.y:.4f}, z={pos.z:.4f}")
            print(f"朝向:  qx={ori.x:.4f}, qy={ori.y:.4f}, "
                  f"qz={ori.z:.4f}, qw={ori.w:.4f}")
            print(f"定位状态: {odom.loc_state}, 置信度: {odom.loc_confidence:.3f}")
            print()
            print("复制以下内容添加到 WAYPOINTS 字典:")
            print(f'    "新导航点": {{')
            print(f'        "position": [{pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f}],')
            print(f'        "orientation": [{ori.x:.4f}, {ori.y:.4f}, '
                  f'{ori.z:.4f}, {ori.w:.4f}],')
            print(f'    }},')
        except Exception as e:
            print(f"获取当前位置失败: {e}")

    def navigate_to(self, name: str, high_precision: bool = False,
                    timeout: float = 120.0) -> bool:
        """
        导航到指定名称的导航点
        
        Args:
            name: 导航点名称（对应WAYPOINTS字典的键）
            high_precision: True=高精度导航，False=普通导航
            timeout: 超时时间（秒）
        Returns:
            True=成功到达, False=失败/超时
        """
        if name not in WAYPOINTS:
            print(f"❌ 导航点 '{name}' 不存在")
            print(f"   可用导航点: {list(WAYPOINTS.keys())}")
            return False

        wp = WAYPOINTS[name]
        pos = wp["position"]
        ori = wp["orientation"]

        print(f"\n🚀 开始导航到: '{name}'")
        print(f"   目标位置: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        print(f"   目标朝向: ({ori[0]:.3f}, {ori[1]:.3f}, {ori[2]:.3f}, {ori[3]:.3f})")
        print(f"   导航模式: {'高精度' if high_precision else '普通'}")

        # 构建导航请求
        target = agibot_gdk.NaviReq()
        target.target.position.x = pos[0]
        target.target.position.y = pos[1]
        target.target.position.z = pos[2]
        target.target.orientation.x = ori[0]
        target.target.orientation.y = ori[1]
        target.target.orientation.z = ori[2]
        target.target.orientation.w = ori[3]

        # 发送导航命令
        try:
            if high_precision:
                self.pnc.high_precision_navi(target)
            else:
                self.pnc.normal_navi(target)
            print("   ✅ 导航命令已发送，等待到达...")
        except Exception as e:
            print(f"   ❌ 发送导航命令失败: {e}")
            return False

        # 等待导航完成
        return self._wait_navigation(name, timeout)

    def _wait_navigation(self, target_name: str, timeout: float) -> bool:
        """轮询等待导航任务完成"""
        start_time = time.time()
        last_state = -1
        dot_count = 0

        while time.time() - start_time < timeout:
            try:
                task = self.pnc.get_task_state()
                state = task.state
                state_name = self.STATE_NAMES.get(state, f"未知({state})")

                # 状态变化时打印
                if state != last_state:
                    print(f"\n   状态变化: {state_name}")
                    last_state = state

                # 成功
                if state == self.STATE_SUCCESS:
                    elapsed = time.time() - start_time
                    print(f"\n✅ 已到达 '{target_name}'！耗时 {elapsed:.1f} 秒")
                    return True

                # 失败
                if state in (self.STATE_FAILED, self.STATE_CANCELED):
                    print(f"\n❌ 导航到 '{target_name}' 失败，状态: {state_name}")
                    if task.message:
                        print(f"   原因: {task.message}")
                    return False

                # 运行中：打印进度点
                if state == self.STATE_RUNNING:
                    dot_count += 1
                    elapsed = time.time() - start_time
                    print(f"\r   导航中... {elapsed:.0f}s / {timeout:.0f}s   ", end="")

            except Exception as e:
                print(f"\n   获取任务状态失败: {e}")

            time.sleep(0.5)

        print(f"\n⏰ 导航超时（{timeout}秒）")
        # 超时后取消任务
        try:
            ts = self.pnc.get_task_state()
            self.pnc.cancel_task(ts.id)
            print("   已发送取消命令")
        except Exception:
            pass
        return False

    def navigate_sequence(self, waypoint_names: list,
                          high_precision: bool = False,
                          stop_on_fail: bool = True) -> bool:
        """
        依次导航到多个导航点

        Args:
            waypoint_names: 导航点名称列表
            high_precision: 是否使用高精度导航
            stop_on_fail: 失败时是否停止后续导航
        Returns:
            True=全部成功, False=有失败
        """
        print(f"\n📍 开始序列导航，共 {len(waypoint_names)} 个点")
        print(f"   顺序: {' → '.join(waypoint_names)}")

        all_success = True
        for i, name in enumerate(waypoint_names):
            print(f"\n[{i+1}/{len(waypoint_names)}] 前往: '{name}'")
            success = self.navigate_to(name, high_precision)

            if not success:
                all_success = False
                if stop_on_fail:
                    print(f"\n⛔ 导航到 '{name}' 失败，停止序列")
                    break
                else:
                    print(f"\n⚠️  导航到 '{name}' 失败，继续下一个")

            if i < len(waypoint_names) - 1:
                print("   等待 1 秒后前往下一个导航点...")
                time.sleep(1.0)

        if all_success:
            print("\n🎉 序列导航全部完成！")
        else:
            print("\n⚠️  序列导航未完全成功")
        return all_success

    def cancel_current_task(self):
        """取消当前导航任务"""
        try:
            ts = self.pnc.get_task_state()
            state_name = self.STATE_NAMES.get(ts.state, str(ts.state))
            print(f"当前任务状态: {state_name} (ID: {ts.id})")
            if ts.state in (self.STATE_RUNNING, self.STATE_PAUSED, self.STATE_STARTING):
                self.pnc.cancel_task(ts.id)
                print("✅ 已取消任务")
            else:
                print("当前无运行中的任务")
        except Exception as e:
            print(f"取消任务失败: {e}")

    def interactive_menu(self):
        """交互式菜单"""
        while True:
            print("\n" + "="*50)
            print("G2 导航点菜单")
            print("="*50)
            print("0. 查看机器人当前位置（用于记录新导航点）")
            print("1. 查看地图信息")
            wp_list = list(WAYPOINTS.keys())
            for i, name in enumerate(wp_list):
                pos = WAYPOINTS[name]["position"]
                print(f"{i+2}. 导航到: {name}  [{pos[0]:.2f}, {pos[1]:.2f}]")
            print(f"{len(wp_list)+2}. 取消当前任务")
            print("q. 退出")
            print("-"*50)

            choice = input("请选择: ").strip().lower()

            if choice == 'q':
                break
            elif choice == '0':
                self.show_current_pose()
            elif choice == '1':
                self.show_map_info()
            elif choice == str(len(wp_list)+2):
                self.cancel_current_task()
            else:
                try:
                    idx = int(choice) - 2
                    if 0 <= idx < len(wp_list):
                        wp_name = wp_list[idx]
                        mode = input("导航模式? (1=普通[默认], 2=高精度): ").strip()
                        high_prec = (mode == '2')
                        self.navigate_to(wp_name, high_precision=high_prec)
                    else:
                        print("无效选择")
                except ValueError:
                    print("无效输入")


# ═══════════════════════════════════════════════════════════════
#  命令行入口
# ═══════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser(
        description="G2机器人 导航到地图导航点",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python3 navi_to_waypoint.py              # 交互式菜单
  python3 navi_to_waypoint.py --list       # 列出导航点
  python3 navi_to_waypoint.py --pose       # 显示当前位置（记录新导航点用）
  python3 navi_to_waypoint.py --wp A点     # 直接导航到"A点"
  python3 navi_to_waypoint.py --wp 0       # 导航到第0号导航点（按索引）
  python3 navi_to_waypoint.py --wp A点 --high-precision  # 高精度导航
  python3 navi_to_waypoint.py --seq A点 B点 充电桩       # 依次导航到多点
  python3 navi_to_waypoint.py --timeout 60 --wp A点      # 60秒超时
        """
    )
    ap.add_argument("--wp",    type=str, default=None,
                    help="目标导航点名称或索引（从0开始）")
    ap.add_argument("--seq",   type=str, nargs="+", default=None,
                    help="序列导航点名称列表")
    ap.add_argument("--list",  action="store_true",
                    help="列出所有导航点")
    ap.add_argument("--pose",  action="store_true",
                    help="显示机器人当前位置（用于记录导航点坐标）")
    ap.add_argument("--high-precision", action="store_true",
                    help="使用高精度导航模式")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="导航超时时间（秒，默认120）")
    return ap.parse_args()


def resolve_waypoint_name(key: str) -> Optional[str]:
    """将名称或索引转换为实际名称"""
    wp_list = list(WAYPOINTS.keys())
    # 先按名称查找
    if key in WAYPOINTS:
        return key
    # 再按索引查找
    try:
        idx = int(key)
        if 0 <= idx < len(wp_list):
            return wp_list[idx]
    except ValueError:
        pass
    return None


def main():
    args = parse_args()

    # 仅列出导航点，不需要初始化GDK
    if args.list:
        print("\n已配置的导航点：")
        print("-" * 60)
        for i, (name, data) in enumerate(WAYPOINTS.items()):
            pos = data["position"]
            ori = data["orientation"]
            print(f"[{i}] {name}")
            print(f"     位置:  ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            print(f"     朝向:  ({ori[0]:.3f}, {ori[1]:.3f}, {ori[2]:.3f}, {ori[3]:.3f})")
        print("-" * 60)
        print(f"共 {len(WAYPOINTS)} 个导航点")
        print("\n⚠️  使用前请先在G02 Pad上完成重定位！")
        return

    # 初始化
    nav = WaypointNavigator()

    if args.pose:
        nav.show_current_pose()
        return

    if args.wp is not None:
        name = resolve_waypoint_name(args.wp)
        if name is None:
            print(f"❌ 找不到导航点: '{args.wp}'")
            print(f"   可用: {list(WAYPOINTS.keys())}")
            sys.exit(1)
        success = nav.navigate_to(name, args.high_precision, args.timeout)
        sys.exit(0 if success else 1)

    if args.seq is not None:
        names = []
        for key in args.seq:
            name = resolve_waypoint_name(key)
            if name is None:
                print(f"❌ 找不到导航点: '{key}'")
                sys.exit(1)
            names.append(name)
        success = nav.navigate_sequence(names, args.high_precision)
        sys.exit(0 if success else 1)

    # 默认：交互式菜单
    nav.show_map_info()
    print("\n⚠️  重要：请确保已在G02 Pad上完成重定位，否则导航可能偏差或失败！")
    nav.interactive_menu()


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════
#  获取导航点坐标的方法（重要！）
# ═══════════════════════════════════════════════════════════════
#
# 方法1：用本脚本记录（推荐）
#   1. 先在Pad上完成重定位，确保机器人知道自己在地图上的位置
#   2. 手动推动/遥控机器人到你想设置为导航点的位置
#   3. 运行：python3 navi_to_waypoint.py --pose
#   4. 把输出的坐标复制到上面的 WAYPOINTS 字典
#
# 方法2：从Pad的地图界面读取
#   在Pad的地图界面，长按目标位置，可以查看该位置的地图坐标
#
# 方法3：参考 pnc_example.py 中的坐标
#   官方示例中提供了一些测试坐标：
#   x=9.1016431138415435, y=-4.1492533213758431, z=-0.056417739896994348
#   qx=-0.00075885, qy=0.00151116, qz=-0.61837471, qw=0.78588157
#
# 注意事项：
#   - 每次建新地图，坐标体系会重置，需要重新记录导航点
#   - 导航前必须先在Pad上重定位，GDK代码无法自动完成重定位
#   - 如果机器人在导航中途被人推动，需要重新重定位
# ═══════════════════════════════════════════════════════════════