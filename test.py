import agibot_gdk, time

agibot_gdk.gdk_init()
map_mgr = agibot_gdk.Map()
slam = agibot_gdk.Slam()
time.sleep(2)

# 获取当前地图 ID
curr = map_mgr.get_curr_map()
print(f"当前地图 ID: {curr.id}, 名称: {curr.name}")
if curr.id != 12:
    try:
        map_mgr.switch_map(12)
        print("已切换到地图 12")
    except Exception as e:
        print(f"切换失败: {e}")
time.sleep(1)

# 等待定位成功
print("等待 SLAM 定位...")
for i in range(20):
    try:
        odom = slam.get_odom_info()
        if odom and odom.pose:
            print(f"定位成功！位置: ({odom.pose.pose.position.x:.3f}, {odom.pose.pose.position.y:.3f})")
            break
        else:
            print(f"等待中... (i={i})")
    except Exception as e:
        print(f"获取里程计失败: {e}")
    time.sleep(1)
else:
    print("定位超时，SLAM 仍无输出")

agibot_gdk.gdk_release()