#!/usr/bin/env python3
# coding=utf-8
import cv2
import pygame
import subprocess
import sys
import time
import os
from TransbotCamera import TransbotCamera


class FaceDetector:
    """人脸检测器类"""
    
    def __init__(self, debug=False):
        """
        初始化人脸检测器
        :param debug: 是否开启调试模式
        """
        self.debug = debug
        self.camera = TransbotCamera(debug=debug)
        
        # 加载预训练的人脸检测模型
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # 检查模型是否加载成功
        if self.face_cascade.empty():
            if self.debug:
                print("人脸检测模型加载失败")
            raise Exception("人脸检测模型加载失败")
        
        # 人脸检测参数
        self.scale_factor = 1.1  # 图像缩放因子
        self.min_neighbors = 5   # 最小邻居数
        self.min_size = (60, 60)  # 最小人脸尺寸
        self.max_size = None      # 最大人脸尺寸（空表示不限制）
        
        # 音频相关
        self.host_audio_file = "welcome_host.mp3"  # 指定用户音频文件
        self.guest_audio_file = "welcome_guest.mp3"  # 非指定用户音频文件
        self.audio_timeout = 3000        # 音频播放超时时间(毫秒)
        self.pygame_initialized = False
        
        # 人脸检测状态
        self.face_detected = False
        self.detection_count = 0
        self.required_detections = 5  # 连续检测到多少次才认为有效
        
        # 下一个程序设置
        self.next_program = "ArrowMatching.py"  # 默认下一个程序
        
        if self.debug:
            print("人脸检测器初始化完成")
    
    def initialize_pygame(self):
        """初始化pygame音频系统"""
        try:
            if not self.pygame_initialized:
                pygame.mixer.init()
                self.pygame_initialized = True
                if self.debug:
                    print("pygame音频系统初始化完成")
            return True
        except Exception as e:
            if self.debug:
                print(f"pygame初始化失败: {e}")
            return False
    
    def detect_faces(self, frame):
        """
        在给定帧中检测人脸
        :param frame: 输入图像帧
        :return: 检测到的人脸列表 [(x, y, w, h), ...]
        """
        try:
            # 转换为灰度图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 高斯模糊降噪
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 构建检测参数
            detect_params = {
                'scaleFactor': self.scale_factor,
                'minNeighbors': self.min_neighbors,
                'minSize': self.min_size
            }
            
            # 只有当 maxSize 不为 None 且格式正确时才添加
            if self.max_size is not None and len(self.max_size) == 2:
                detect_params['maxSize'] = self.max_size
            
            # 检测人脸
            faces = self.face_cascade.detectMultiScale(gray, **detect_params)
            
            return faces
            
        except Exception as e:
            if self.debug:
                print(f"人脸检测失败: {e}")
            return []
    
    def draw_face_rectangles(self, frame, faces):
        """
        在图像上绘制人脸矩形框
        :param frame: 输入图像帧
        :param faces: 人脸列表
        :return: 绘制后的图像帧
        """
        try:
            for (x, y, w, h) in faces:
                # 绘制蓝色矩形框
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # 添加文字标签
                cv2.putText(frame, "Face", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            return frame
            
        except Exception as e:
            if self.debug:
                print(f"绘制人脸框失败: {e}")
            return frame
    
    def play_welcome_audio(self, is_host=True):
        """
        播放欢迎音频
        :param is_host: 是否为主人
        :return: 播放是否成功
        """
        try:
            if not self.pygame_initialized:
                if not self.initialize_pygame():
                    return False
            
            # 根据是否为主人选择音频文件
            audio_file = self.host_audio_file if is_host else self.guest_audio_file
            
            # 检查音频文件是否存在
            if not os.path.exists(audio_file):
                if self.debug:
                    print(f"音频文件不存在: {audio_file}")
                return False
            
            # 加载并播放音频
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            if self.debug:
                user_type = "主人" if is_host else "客人"
                print(f"正在播放{user_type}欢迎音频: {audio_file}")
            
            # 等待音频播放完成
            wait_time = 0
            while pygame.mixer.music.get_busy() and wait_time < self.audio_timeout:
                pygame.time.Clock().tick(10)
                wait_time += 10
            
            if self.debug:
                print("欢迎音频播放完成")
            
            return True
            
        except Exception as e:
            if self.debug:
                print(f"播放欢迎音频失败: {e}")
            return False
    
    def launch_next_program(self):
        """启动下一个程序"""
        try:
            if self.debug:
                print(f"启动下一个程序: {self.next_program}")
            
            # 启动新的子进程
            subprocess.Popen(["python3", self.next_program])
            
            if self.debug:
                print(f"程序 {self.next_program} 启动成功")
            
            return True
            
        except Exception as e:
            if self.debug:
                print(f"启动程序失败: {e}")
            return False
    
    def initialize(self):
        """
        初始化人脸检测器
        :return: 初始化是否成功
        """
        try:
            if self.debug:
                print("初始化人脸检测器...")
            
            # 初始化摄像头
            if not self.camera.initialize():
                if self.debug:
                    print("摄像头初始化失败")
                return False
            
            self.camera.move_to_preset('face')

            # 初始化pygame音频系统
            self.initialize_pygame()
            
            # 重置检测状态
            self.reset_detection_state()
            
            if self.debug:
                print("人脸检测器初始化成功")
            
            return True
            
        except Exception as e:
            if self.debug:
                print(f"初始化失败: {e}")
            return False
    
    def reset_detection_state(self):
        """重置检测状态"""
        self.face_detected = False
        self.detection_count = 0
        
        if self.debug:
            print("检测状态已重置")
    
    def detect_and_process(self, show_window=True, is_host=True):
        """
        主要的人脸检测和处理循环
        :param show_window: 是否显示检测窗口
        :param is_host: 是否为主人（True=主人，False=客人）
        :return: 是否成功检测到人脸并处理
        """
        if self.debug:
            user_type = "主人" if is_host else "客人"
            print(f"开始人脸检测... (识别为{user_type})")
        
        try:
            while self.camera.is_camera_opened():
                # 获取摄像头帧
                ret, frame = self.camera.get_camera_frame()
                if not ret:
                    if self.debug:
                        print("无法获取摄像头帧")
                    continue
                
                # 检测人脸
                faces = self.detect_faces(frame)
                
                if len(faces) > 0:
                    # 检测到人脸
                    self.detection_count += 1
                    
                    if self.debug:
                        print(f"检测到 {len(faces)} 个人脸 (第 {self.detection_count} 次)")
                    
                    # 绘制人脸框
                    frame = self.draw_face_rectangles(frame, faces)
                    
                    # 连续检测到足够次数，认为检测成功
                    if self.detection_count >= self.required_detections:
                        self.face_detected = True
                        
                        if self.debug:
                            user_type = "主人" if is_host else "客人"
                            print(f"人脸检测成功！识别为{user_type}")
                        
                        # 播放对应的欢迎音频
                        if self.play_welcome_audio(is_host):
                            if self.debug:
                                print("欢迎音频播放完成")
                        else:
                            if self.debug:
                                print("欢迎音频播放失败")
                    
                        if self.debug:
                            print("人脸检测和音频播放完成，程序结束")
                            return True  # 返回成功并结束程序
                        
                        # 关闭显示窗口
                        if show_window:
                            cv2.destroyAllWindows()
                
                else:
                    # 没有检测到人脸，重置计数
                    self.detection_count = 0
                
                # 显示检测窗口
                if show_window:
                    # 添加状态信息
                    user_type = "主人" if is_host else "客人"
                    cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Count: {self.detection_count}/{self.required_detections}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Mode: {user_type}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow('Face Detection', frame)
                    
                    # 检查按键
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        if self.debug:
                            print("用户按下 'q' 键退出")
                        break
                    elif key == 27:  # ESC键
                        if self.debug:
                            print("用户按下 ESC 键退出")
                        break
                    elif key == ord('h'):  # 按'h'键切换为主人模式
                        is_host = True
                        if self.debug:
                            print("切换为主人模式")
                    elif key == ord('g'):  # 按'g'键切换为客人模式
                        is_host = False
                        if self.debug:
                            print("切换为客人模式")
            
            return False
            
        except KeyboardInterrupt:
            if self.debug:
                print("程序被用户中断")
            return False
        except Exception as e:
            if self.debug:
                print(f"检测过程中发生错误: {e}")
            return False
    
    def set_detection_parameters(self, scale_factor=None, min_neighbors=None, 
                                min_size=None, max_size=None):
        """
        设置人脸检测参数
        :param scale_factor: 图像缩放因子
        :param min_neighbors: 最小邻居数
        :param min_size: 最小人脸尺寸
        :param max_size: 最大人脸尺寸
        """
        if scale_factor is not None:
            self.scale_factor = scale_factor
        if min_neighbors is not None:
            self.min_neighbors = min_neighbors
        if min_size is not None:
            self.min_size = min_size
        if max_size is not None:
            self.max_size = max_size
        
        if self.debug:
            print(f"检测参数更新: scale_factor={self.scale_factor}, "
                  f"min_neighbors={self.min_neighbors}, "
                  f"min_size={self.min_size}, max_size={self.max_size}")
    
    def set_audio_settings(self, host_audio_file=None, guest_audio_file=None, timeout=None):
        """
        设置音频相关参数
        :param host_audio_file: 主人音频文件路径
        :param guest_audio_file: 客人音频文件路径
        :param timeout: 播放超时时间(毫秒)
        """
        if host_audio_file is not None:
            self.host_audio_file = host_audio_file
        if guest_audio_file is not None:
            self.guest_audio_file = guest_audio_file
        if timeout is not None:
            self.audio_timeout = timeout
    
    def set_next_program(self, program_path):
        """
        设置下一个要启动的程序
        :param program_path: 程序路径
        """
        self.next_program = program_path
        
        if self.debug:
            print(f"下一个程序设置为: {self.next_program}")
    
    def print_status(self):
        """打印当前状态"""
        print(f"\n人脸检测器状态:")
        print(f"  检测状态: {'已检测到人脸' if self.face_detected else '等待检测'}")
        print(f"  检测次数: {self.detection_count}/{self.required_detections}")
        print(f"  检测参数: scale_factor={self.scale_factor}, min_neighbors={self.min_neighbors}")
        print(f"  最小尺寸: {self.min_size}")
        print(f"  主人音频: {self.host_audio_file}")
        print(f"  客人音频: {self.guest_audio_file}")
        print(f"  下一个程序: {self.next_program}")
        print(f"  pygame状态: {'已初始化' if self.pygame_initialized else '未初始化'}")
        self.camera.print_status()
    
    def close(self):
        """关闭人脸检测器"""
        if self.debug:
            print("关闭人脸检测器...")
        
        # 关闭摄像头
        self.camera.close()
        
        # 关闭pygame
        if self.pygame_initialized:
            pygame.mixer.quit()
            self.pygame_initialized = False
        
        # 关闭OpenCV窗口
        cv2.destroyAllWindows()
        
        if self.debug:
            print("人脸检测器已关闭")


def Face():
    """
    兼容性函数：保持原有的调用方式
    :return: 人脸检测是否成功
    """
    detector = FaceDetector(debug=False)
    
    try:
        # 初始化检测器
        if not detector.initialize():
            print("人脸检测器初始化失败")
            return False
        
        # 开始检测
        result = detector.detect_and_process(show_window=False)
        
        return result
    
    except Exception as e:
        print(f"人脸检测失败: {e}")
        return False
    
    finally:
        detector.close()


def test_face_detection():
    """测试人脸检测功能"""
    detector = FaceDetector(debug=True)
    
    try:
        # 初始化
        if not detector.initialize():
            print("初始化失败")
            return
        
        print("人脸检测测试开始...")
        print("请面向摄像头，按 'q' 或 ESC 退出测试")
        
        # 开始检测（显示窗口）
        result = detector.detect_and_process(show_window=True, is_host=False)
        
        if result:
            print("人脸检测成功！")
        else:
            print("人脸检测失败或被用户中断")
        
    except Exception as e:
        print(f"测试失败: {e}")
    finally:
        detector.close()


if __name__ == "__main__":
    # 选择测试模式
    print("选择测试模式:")
    print("1. 兼容性测试 (使用 Face 函数)")
    print("2. 类测试 (使用 FaceDetector 类)")
    print("3. 交互式测试 (显示检测窗口)")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        # 兼容性测试
        print("兼容性测试开始...")
        result = Face()
        print(f"检测结果: {'成功' if result else '失败'}")
        
    elif choice == "2":
        # 类测试
        detector = FaceDetector(debug=True)
        try:
            detector.initialize()
            result = detector.detect_and_process(show_window=False, is_host=True)
            print(f"检测结果: {'成功' if result else '失败'}")
        finally:
            detector.close()
            
    elif choice == "3":
        # 交互式测试
        test_face_detection()
        
    else:
        print("无效选择")