#!/usr/bin/env python3
import agibot_gdk, time

agibot_gdk.gdk_init()
pnc = agibot_gdk.Pnc()
time.sleep(1.5)

# 查看当前任务状态
ts = pnc.get_task_state()
print(f"当前任务状态: {ts.state}, ID: {ts.id}")

# 只在有运行中任务时才取消
if ts.state in (1, 2, 3, 4, 5):
    pnc.cancel_task(ts.id)
    print("已取消旧任务")
    time.sleep(0.5)

# 相对移动：向前走0.5米
target = agibot_gdk.NaviReq()
target.target.position.x = 0.5
target.target.position.y = 0.0
target.target.position.z = 0.0
target.target.orientation.x = 0.0
target.target.orientation.y = 0.0
target.target.orientation.z = 0.0
target.target.orientation.w = 1.0

try:
    pnc.relative_move(target)
    print("相对移动命令已发送，等待结果...")
except Exception as e:
    print(f"发送失败: {e}")
    exit(1)

# 等待结果
for i in range(30):
    time.sleep(0.5)
    ts = pnc.get_task_state()
    states = {0:"空闲",1:"启动中",2:"运行中",3:"暂停中",4:"已暂停",
              5:"恢复中",6:"取消中",7:"已取消",8:"失败",9:"成功"}
    print(f"状态: {states.get(ts.state, ts.state)}  消息: {ts.message}")
    if ts.state in (7, 8, 9):
        break