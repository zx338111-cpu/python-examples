import agibot_gdk

if __name__ == "__main__":
    # 初始化 GDK
    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        print("GDK 初始化失败")
        exit(1)
    print("GDK 初始化成功")

    # 创建地图管理器对象
    map_manager = agibot_gdk.Map()
    # 等待地图管理器初始化，这里等待2秒
    # 根据之前的讨论，这样能确保系统稳定
    import time
    time.sleep(2)

    # 获取所有地图
    all_maps = map_manager.get_all_map()
    print(f"找到 {len(all_maps)} 张地图:")
    for i, map_name in enumerate(all_maps):
        # 打印地图ID、名称，并标记是否为当前使用的地图
        current_marker = " [当前使用]" if map_name.is_curr_map else ""
        print(f"  {i+1}. ID: {map_name.id}, 名称: {map_name.name}{current_marker}")

    # 释放 GDK 资源
    agibot_gdk.gdk_release()
    print("GDK 资源已释放")