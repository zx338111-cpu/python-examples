#!/usr/bin/env python3
"""
交互(Interaction)接口使用演示程序
展示如何使用agibot_gdk进行语音交互、显示控制、音频播放等功能
"""

import time
import agibot_gdk
import sys
import select
import threading
import json
from typing import Optional


class InteractionDemo:
    def __init__(self):
        """初始化交互对象"""
        print("正在初始化交互对象...")
        self.interaction = agibot_gdk.Interaction()
        time.sleep(1)  # 等待初始化完成
        print("交互对象初始化完成！")

    def test_set_language(self):
        """测试设置语言"""
        print(f"\n{'='*60}")
        print("测试 set_language() - 设置语音语言")
        print(f"{'='*60}")
        
        try:
            # 测试设置中文
            print("设置语言为中文...")
            self.interaction.set_language(agibot_gdk.Language.kLanguageChinese)
            print("设置中文成功")
            
            time.sleep(0.5)
            
            # 测试设置英文
            print("设置语言为英文...")
            self.interaction.set_language(agibot_gdk.Language.kLanguageEnglish)
            print("设置英文成功")
            
            # 恢复为中文
            self.interaction.set_language(agibot_gdk.Language.kLanguageChinese)
            return True
        except Exception as e:
            print(f"设置语言失败: {e}")
            return False

    def test_set_volume(self):
        """测试设置音量"""
        print(f"\n{'='*60}")
        print("测试 set_volume() - 设置音量")
        print(f"{'='*60}")
        
        try:
            # 测试不同音量值
            volumes = [30, 50, 70]
            for volume in volumes:
                print(f"设置音量为 {volume}...")
                self.interaction.set_volume(volume)
                print(f"设置音量 {volume} 成功")
                time.sleep(0.3)
            
            # 恢复为中等音量
            self.interaction.set_volume(50)
            return True
        except Exception as e:
            print(f"设置音量失败: {e}")
            return False

    def test_set_wakeup_switch(self):
        """测试设置唤醒开关"""
        print(f"\n{'='*60}")
        print("测试 set_wakeup_switch() - 设置唤醒开关")
        print(f"{'='*60}")
        
        try:
            # 开启唤醒
            print("开启唤醒功能...")
            self.interaction.set_wakeup_switch(True)
            print("开启唤醒功能成功")
            
            time.sleep(0.5)
            
            # 关闭唤醒
            print("关闭唤醒功能...")
            self.interaction.set_wakeup_switch(False)
            print("关闭唤醒功能成功")
            
            # 恢复为开启状态
            self.interaction.set_wakeup_switch(True)
            return True
        except Exception as e:
            print(f"设置唤醒开关失败: {e}")
            return False

    def test_set_audio_switch(self):
        """测试设置音频开关"""
        print(f"\n{'='*60}")
        print("测试 set_audio_switch() - 设置音频开关")
        print(f"{'='*60}")
        
        try:
            # 开启音频
            print("开启音频功能...")
            self.interaction.set_audio_switch(True)
            print("开启音频功能成功")
            
            time.sleep(0.5)
            
            # 关闭音频
            print("关闭音频功能...")
            self.interaction.set_audio_switch(False)
            print("关闭音频功能成功")
            
            # 恢复为开启状态
            self.interaction.set_audio_switch(True)
            return True
        except Exception as e:
            print(f"设置音频开关失败: {e}")
            return False

    def test_set_display_switch(self):
        """测试设置显示开关"""
        print(f"\n{'='*60}")
        print("测试 set_display_switch() - 设置显示开关")
        print(f"{'='*60}")
        
        try:
            # 开启显示
            print("开启显示功能...")
            self.interaction.set_display_switch(True)
            print("开启显示功能成功")
            
            time.sleep(0.5)
            
            # 关闭显示
            print("关闭显示功能...")
            self.interaction.set_display_switch(False)
            print("关闭显示功能成功")
            
            # 恢复为开启状态
            self.interaction.set_display_switch(True)
            return True
        except Exception as e:
            print(f"设置显示开关失败: {e}")
            return False

    def test_play_tts(self):
        """测试播放TTS"""
        print(f"\n{'='*60}")
        print("测试 play_tts() - 播放TTS（文本转语音）")
        print(f"{'='*60}")
        
        try:
            # 确保语言设置为中文
            self.interaction.set_language(agibot_gdk.Language.kLanguageChinese)
            time.sleep(2)
            
            # 播放中文TTS
            test_texts = [
                "你好，我是精灵G2机器人"
            ]
            
            for text in test_texts:
                print(f"播放TTS: {text}")
                self.interaction.play_tts(text)
                print("TTS播放命令已发送")
                time.sleep(2)  # 等待播放完成
            
            return True
        except Exception as e:
            print(f"播放TTS失败: {e}")
            return False

    def test_play_audio(self):
        """测试播放音频文件"""
        print(f"\n{'='*60}")
        print("测试 play_audio() - 播放音频文件")
        print(f"{'='*60}")
        
        try:
            # 注意：这里使用示例路径，实际使用时需要提供真实路径
            audio_path = "/path/to/audio.wav"
            print(f"尝试播放音频文件: {audio_path}")
            print("注意: 如果文件不存在，此测试会失败")
            
            # 实际使用时取消注释
            # self.interaction.play_audio(audio_path)
            # print("音频播放命令已发送")
            # time.sleep(5)  # 等待播放完成
            
            print("音频播放测试跳过（需要提供真实文件路径）")
            return True
        except Exception as e:
            print(f"播放音频失败: {e}")
            return False

    def test_play_video(self):
        """测试播放视频文件"""
        print(f"\n{'='*60}")
        print("测试 play_video() - 播放视频文件")
        print(f"{'='*60}")
        
        try:
            # 注意：这里使用示例路径，实际使用时需要提供真实路径
            video_path = "/path/to/video.mp4"
            loop_count = 1
            print(f"尝试播放视频文件: {video_path}")
            print(f"循环次数: {loop_count}")
            print("注意: 如果文件不存在，此测试会失败")
            
            # 实际使用时取消注释
            # self.interaction.play_video(video_path, loop_count)
            # print("视频播放命令已发送")
            # time.sleep(10)  # 等待播放完成
            
            print("视频播放测试跳过（需要提供真实文件路径）")
            return True
        except Exception as e:
            print(f"播放视频失败: {e}")
            return False

    def test_get_func_status(self):
        """测试获取功能状态"""
        print(f"\n{'='*60}")
        print("测试 get_func_status() - 获取语音功能状态")
        print(f"{'='*60}")
        
        try:
            func_status = self.interaction.get_func_status()
            
            print("功能状态信息:")
            print(f"  功能状态: {func_status.func_status}")
            print(f"  唤醒状态: {func_status.wakeup_status}")
            print(f"  请求者: {func_status.requester}")
            print(f"  唤醒功能启用: {func_status.wakeup_enabled}")
            print(f"  显示功能启用: {func_status.display_enabled}")
            print(f"  音频功能启用: {func_status.audio_enabled}")
            
            print(f"\n中文语音设置:")
            print(f"  音量: {func_status.cn_settings.volume}")
            print(f"  语速: {func_status.cn_settings.speech_rate}")
            print(f"  音色: {func_status.cn_settings.voice_tone}")
            print(f"  是否为当前设置: {func_status.cn_settings.is_curr_setting}")
            
            print(f"\n英文语音设置:")
            print(f"  音量: {func_status.en_settings.volume}")
            print(f"  语速: {func_status.en_settings.speech_rate}")
            print(f"  音色: {func_status.en_settings.voice_tone}")
            print(f"  是否为当前设置: {func_status.en_settings.is_curr_setting}")
            
            print(f"\n时间戳: {func_status.timestamp}")
            return True
        except Exception as e:
            print(f"获取功能状态失败: {e}")
            return False

    def test_get_asr_text(self):
        """测试获取ASR文本"""
        print(f"\n{'='*60}")
        print("测试 get_asr_text() - 获取ASR（自动语音识别）文本")
        print(f"{'='*60}")
        
        try:
            print("开启通话模式...")
            print("注意: 需要确保语音识别功能已启用(对机器人说 \"你好，精灵\" )，并且有语音输入")
            self.interaction.set_call_mode(True)
            
            # 创建ASR处理器并注册回调函数
            asr_handler = ASRHandler(self.interaction)
            self.interaction.register_callback("get_asr_text", asr_handler.callback)
            print("回调函数已注册")
            
            time.sleep(1)
            
            print("\n" + "="*60)
            print("   - 可以通过语音命令退出：说'退出'、'结束'、'结束测试'")
            print("   - 可以通过键盘输入退出：输入 'q' 或 'Q' 后按回车")
            print("   - 可以通过 Ctrl+C 退出")
            print("="*60 + "\n")
            
            print("开始监听语音输入（回调函数会自动处理）...")
            
            # 使用非阻塞方式检测键盘输入
            def check_keyboard_input():
                while not asr_handler.get_should_exit():
                    try:
                        if sys.stdin.isatty():
                            if select.select([sys.stdin], [], [], 0.1)[0]:
                                user_input = sys.stdin.readline().strip().lower()
                                if user_input == 'q':
                                    print("\n[键盘输入] 收到退出命令 'q'")
                                    asr_handler.set_should_exit(True)
                                    break
                    except Exception as e:
                        print(f"键盘输入检测出错: {e}")
                    time.sleep(0.1)
            
            # 启动键盘监听线程
            keyboard_thread = threading.Thread(target=check_keyboard_input, daemon=True)
            keyboard_thread.start()
            
            # 主循环：持续等待，直到收到退出命令
            # 回调函数在后台持续工作，每次识别到新文本都会自动调用
            try:
                while not asr_handler.get_should_exit():
                    time.sleep(0.1)  # 主循环只需要等待，回调函数会自动处理
            except KeyboardInterrupt:
                print("\n[Ctrl+C] 收到中断信号")
                asr_handler.set_should_exit(True)
            
            # 清理资源
            print("\n正在关闭通话模式...")
            self.interaction.set_call_mode(False)
            self.interaction.unregister_callback("get_asr_text")
            time.sleep(0.5)
            print("测试结束")
            return True
            
        except KeyboardInterrupt:
            print("\n[Ctrl+C] 用户中断测试")
            try:
                self.interaction.set_call_mode(False)
                self.interaction.unregister_callback("get_asr_text")
            except Exception as e:
                print(f"清理资源时出错: {e}")
            return False
        except Exception as e:
            print(f"获取ASR文本失败: {e}")
            print("提示: 如果语音识别功能未启用或没有语音输入，此操作可能失败")
            try:
                self.interaction.set_call_mode(False)
                self.interaction.unregister_callback("get_asr_text")
            except Exception as cleanup_error:
                print(f"清理资源时出错: {cleanup_error}")
            return False

    def get_test_menu(self):
        """获取测试菜单选项"""
        return [
            ("1", "设置语言", self.test_set_language),
            ("2", "设置音量", self.test_set_volume),
            ("3", "设置唤醒开关", self.test_set_wakeup_switch),
            ("4", "设置音频开关", self.test_set_audio_switch),
            ("5", "设置显示开关", self.test_set_display_switch),
            ("6", "获取功能状态", self.test_get_func_status),
            ("7", "播放TTS", self.test_play_tts),
            ("8", "播放音频文件", self.test_play_audio),
            ("9", "播放视频文件", self.test_play_video),
            ("10", "获取ASR文本", self.test_get_asr_text),
            ("0", "运行全部测试", None),
        ]

    def print_menu(self):
        """打印测试菜单"""
        print("\n" + "=" * 60)
        print("交互功能测试菜单")
        print("=" * 60)
        menu = self.get_test_menu()
        for key, name, _ in menu:
            print(f"{key}. {name}")

    def run_single_test(self, test_func, test_name):
        """运行单个测试"""
        try:
            result = test_func()
            status = "成功" if result else "失败"
            print(f"\n{test_name}: {status}")
            return result
        except Exception as e:
            print(f"\n{test_name}执行出错: {e}")
            return False

    def run_comprehensive_test(self):
        """运行综合测试"""
        print("交互功能综合测试程序")
        print("=" * 50)
        
        test_results = []
        
        # 1. 基础设置接口测试
        print("\n1. 基础设置接口测试")
        test_results.append(("设置语言", self.test_set_language()))
        test_results.append(("设置音量", self.test_set_volume()))
        test_results.append(("设置唤醒开关", self.test_set_wakeup_switch()))
        test_results.append(("设置音频开关", self.test_set_audio_switch()))
        test_results.append(("设置显示开关", self.test_set_display_switch()))
        
        # 2. 状态查询接口测试
        print("\n2. 状态查询接口测试")
        test_results.append(("获取功能状态", self.test_get_func_status()))
        
        # 3. 播放接口测试
        print("\n3. 播放接口测试")
        test_results.append(("播放TTS", self.test_play_tts()))
        test_results.append(("播放音频文件", self.test_play_audio()))
        test_results.append(("播放视频文件", self.test_play_video()))
        
        # 4. 语音识别接口测试
        print("\n4. 语音识别接口测试")
        test_results.append(("获取ASR文本", self.test_get_asr_text()))
        
        # 5. 显示测试总结
        print(f"\n{'='*60}")
        print("测试总结")
        print(f"{'='*60}")
        success_count = sum(1 for _, result in test_results if result)
        total_count = len(test_results)
        
        for test_name, result in test_results:
            status = "成功" if result else "失败"
            print(f"{test_name}: {status}")
        
        print(f"\n总测试数: {total_count}")
        print(f"成功数: {success_count}")
        print(f"失败数: {total_count - success_count}")
        print(f"成功率: {success_count/total_count*100:.1f}%")

    def run_interactive_test(self):
        """运行交互式测试"""
        print("交互功能测试程序")
        print("=" * 50)
        
        menu = self.get_test_menu()
        menu_dict = {key: (name, func) for key, name, func in menu}
        
        while True:
            self.print_menu()
            choice = input("\n请选择测试项 (输入数字，q退出): ").strip()
            
            if choice.lower() == 'q':
                print("退出测试程序")
                break
            
            if choice == "0":
                # 运行全部测试
                self.run_comprehensive_test()
            elif choice in menu_dict:
                name, func = menu_dict[choice]
                if func:
                    print(f"\n开始测试: {name}")
                    self.run_single_test(func, name)
                else:
                    self.run_comprehensive_test()
            else:
                print("无效选择，请重新输入")

class ASRHandler:
    def __init__(self, interaction):
        self.interaction = interaction
        self.last_text = ""  # 用于避免重复处理
        self.should_exit = False  # 退出标志
        self._lock = threading.Lock()  # 线程锁，保护共享状态
    
    def callback(self, text):
        """
        ASR文本回调函数
        持续监听并处理语音识别结果，直到收到退出命令
        """
        try:
            # 类型检查和转换
            if not isinstance(text, str):
                text = str(text)
            
            # 过滤空文本
            if not text or text.strip() == "":
                return
            
            # 过滤JSON状态消息（如 {"streamDataType":"done","status":"end"}）
            try:
                data = json.loads(text)
                if isinstance(data, dict) and ("status" in data or "streamDataType" in data):
                    return  # 跳过状态消息
            except (json.JSONDecodeError, ValueError):
                pass  # 不是JSON，继续处理
            
            # 线程安全地检查重复文本
            with self._lock:
                if text == self.last_text:
                    return  # 跳过重复文本
                self.last_text = text
            
            print(f"🎤 识别到: {text}")
            
            # 检查退出命令（优先处理）
            exit_commands = ["退出", "结束", "结束测试", "exit", "quit"]
            if any(cmd in text for cmd in exit_commands):
                print(f"\n[退出命令] 收到: {text}")
                try:
                    self.interaction.play_tts("好的，结束测试")
                except Exception as e:
                    print(f"播放TTS失败: {e}")
                
                with self._lock:
                    self.should_exit = True
                return
            
            # 处理语音命令
            try:
                if text == "你好":
                    self.interaction.play_tts("你好，我是精灵G2机器人")
                elif text == "你是谁":
                    self.interaction.play_tts("我是精灵G2机器人")
                elif text == "你能干什么":
                    self.interaction.play_tts("我能胜任多种工业及工作场景，成为你最贴心的伙伴")
                elif text == "你有二次开发能力吗":
                    self.interaction.play_tts("我有多种语言的开发接口，有丰富的原子能力，可以满足你的定制化开发需求")
                # 其他未匹配的文本不处理，避免干扰
            except Exception as e:
                print(f"处理语音命令时出错: {e}")
                # 异常不会阻止后续回调
            
        except Exception as e:
            # 确保异常不会阻止后续回调
            print(f"[回调异常] {e}")
            import traceback
            traceback.print_exc()
    
    def get_should_exit(self):
        """线程安全地获取退出标志"""
        with self._lock:
            return self.should_exit
    
    def set_should_exit(self, value):
        """线程安全地设置退出标志"""
        with self._lock:
            self.should_exit = value

def main():
    """主函数 - 运行交互式测试"""
    # 初始化GDK系统
    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        print("GDK初始化失败")
        return
    
    print("GDK初始化成功")
    
    try:
        demo = InteractionDemo()
        demo.run_interactive_test()
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 释放GDK系统资源
        if agibot_gdk.gdk_release() != agibot_gdk.GDKRes.kSuccess:
            print("GDK释放失败")
        else:
            print("\nGDK释放成功")


if __name__ == "__main__":
    main()

