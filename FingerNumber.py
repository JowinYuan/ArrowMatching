#!/usr/bin/env python3
# coding=utf-8
import cv2
import time
import mediapipe as mp
import numpy as np
from TransbotCamera import TransbotCamera


class FingerNumberDetector:
    """手势识别检测器"""
    
    def __init__(self, debug=False):
        """
        初始化手势识别检测器
        :param debug: 是否开启调试模式
        """
        self.debug = debug
        self.camera = TransbotCamera(debug=debug)
        
        # MediaPipe初始化
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,  # 最多检测1只手
            min_detection_confidence=0.7,  # 检测置信度阈值（高于0.7才认为是手）
            min_tracking_confidence=0.5  # 跟踪置信度阈值
        )
        # 绘制手部关键点和连接线
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 手势检测状态变量
        self.gesture_detected = False  # 标记是否检测到有效手势
        self.output_value = None  # 存储输出值（1/2/3/4）
        self.prev_pinky_positions = []  # 保存小拇指历史位置，用于挥手检测
        self.frame_count = 0  # 帧计数器，用于控制挥手检测频率
        
        if self.debug:
            print("手势识别检测器初始化完成")
    
    def count_fingers(self, hand_landmarks):
        """
        简化版手指计数（检测食指、中指、无名指、小指）
        :param hand_landmarks: 手部关键点
        :return: 伸出的手指数
        """
        tips = [8, 12, 16, 20]  # 四指指尖的关键点索引
        fingers = []
        for i in range(4):
            tip = hand_landmarks.landmark[tips[i]]  # 指尖坐标
            pip = hand_landmarks.landmark[tips[i] - 2]  # 近指节关节坐标

            # 通过比较Y坐标判断手指是否伸出（Y值越小位置越高）
            fingers.append(1 if tip.y < pip.y else 0)
        return sum(fingers)  # 返回伸出的手指数

    def detect_waving_by_pinky(self, hand_landmarks):
        """
        检测小拇指是否摆动（挥手动作）
        :param hand_landmarks: 手部关键点
        :return: 是否检测到挥手动作
        """
        self.frame_count += 1
        if self.frame_count % 5 != 0:  # 每5帧处理一次
            return False

        # 获取小拇指指尖坐标
        pinky_tip = hand_landmarks.landmark[20]

        # 保存当前位置
        current_pos = (pinky_tip.x, pinky_tip.y)
        self.prev_pinky_positions.append(current_pos)

        # 只保留最近的5个位置
        if len(self.prev_pinky_positions) > 5:
            self.prev_pinky_positions.pop(0)

        if len(self.prev_pinky_positions) == 5:
            x_coords = [pos[0] for pos in self.prev_pinky_positions]
            y_coords = [pos[1] for pos in self.prev_pinky_positions]
            dx = max(x_coords) - min(x_coords)
            dy = max(y_coords) - min(y_coords)
            # 如果位移超过阈值，认为是挥手
            return dx > 0.05 or dy > 0.05
        return False

    def detect_gesture(self):
        """
        手势检测主函数（检测到有效手势后停止）
        :return: 识别到的手指数 (1-4)
        """
        if self.debug:
            print("开始手势检测...")
        
        fingers = 0

        # 持续检测，直到找到有效手势或摄像头关闭
        while fingers == 0 and self.camera.is_camera_opened():
            ret, frame = self.camera.get_camera_frame()
            if not ret:
                if self.debug:
                    print("无法获取摄像头帧")
                continue

            # 图像处理
            frame.flags.writeable = False

            # 将BGR格式（OpenCV默认）转换为RGB模式（MediaPipe要求）
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame.flags.writeable = True

            # 如果检测到手部
            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    # 在图像上绘制手部关键点和连接线（可选）
                    if self.debug:
                        self.mp_drawing.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # 计数手指
                    fingers = self.count_fingers(landmarks)

                    # 检测到1/2/3指时直接输出
                    if fingers in [1, 2, 3]:
                        self.gesture_detected = True
                        self.output_value = fingers
                        if self.debug:
                            print(f"检测到手势: {self.output_value}")
                        break

                    # 4指时检查是否挥手
                    elif fingers == 4:
                        if self.detect_waving_by_pinky(landmarks):
                            self.gesture_detected = True
                            self.output_value = 4
                            if self.debug:
                                print(f"检测到挥手: {self.output_value}")
                            break

            # 显示实时画面（调试模式）
            if self.debug:
                cv2.putText(frame, "请出示手势...", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"当前手指数: {fingers}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Gesture Detection', frame)

                if cv2.waitKey(5) & 0xFF == 27:  # 按ESC退出
                    break
        
        if self.debug:
            print(f"识别到手指数: {fingers}")
        
        return fingers

    def initialize(self):
        """
        初始化检测器
        :return: 初始化是否成功
        """
        if self.debug:
            print("初始化手势识别检测器...")
        
        # 初始化摄像头和硬件
        if not self.camera.initialize():
            if self.debug:
                print("摄像头或硬件初始化失败")
            return False
        
        # 重置检测状态
        self.reset_detection_state()
        
        if self.debug:
            print("手势识别检测器初始化成功")
        
        return True

    def reset_detection_state(self):
        """重置检测状态"""
        self.gesture_detected = False
        self.output_value = None
        self.prev_pinky_positions = []
        self.frame_count = 0
        
        if self.debug:
            print("检测状态已重置")

    def get_finger_number(self):
        """
        获取手指数量的主函数
        :return: 检测到的手指数 (1-4)，失败返回0
        """
        if self.debug:
            print("开始手指数量检测...")
        
        time.sleep(2)  # 等待摄像头稳定
        cnt = 0

        try:
            # 启动手势检测
            cnt = self.detect_gesture()

        except KeyboardInterrupt:
            if self.debug:
                print("程序被用户终止")
        except Exception as e:
            if self.debug:
                print(f"检测过程中发生错误: {e}")
        finally:
            # 确保机器人停止
            if self.camera.bot:
                self.camera.bot.set_car_motion(0, 0)
            
            if self.debug:
                cv2.destroyAllWindows()

        return cnt

    def close(self):
        """关闭检测器"""
        if self.debug:
            print("关闭手势识别检测器...")
        
        # 关闭摄像头和硬件
        self.camera.close()
        
        # 关闭MediaPipe
        if hasattr(self, 'hands'):
            self.hands.close()
        
        # 关闭OpenCV窗口
        cv2.destroyAllWindows()
        
        if self.debug:
            print("手势识别检测器已关闭")

    def print_status(self):
        """打印当前状态"""
        print(f"\n手势识别检测器状态:")
        print(f"  检测状态: {'已检测到手势' if self.gesture_detected else '等待检测'}")
        print(f"  输出值: {self.output_value}")
        print(f"  帧计数: {self.frame_count}")
        print(f"  小拇指位置历史: {len(self.prev_pinky_positions)} 个位置")
        print(f"  摄像头状态: {'已打开' if self.camera.is_camera_opened() else '未打开'}")
        self.camera.print_status()


def FingerNumber():
    """
    兼容性函数：保持原有的函数调用方式
    :return: 检测到的手指数
    """
    detector = FingerNumberDetector(debug=False)
    
    try:
        # 初始化检测器
        if not detector.initialize():
            print("检测器初始化失败")
            return 0
        
        # 获取手指数量
        result = detector.get_finger_number()
        
        return result
    
    except Exception as e:
        print(f"手势检测失败: {e}")
        return 0
    
    finally:
        detector.close()


# 测试函数
def test_finger_detection():
    """测试手势检测功能"""
    detector = FingerNumberDetector(debug=True)
    
    try:
        # 初始化
        if not detector.initialize():
            print("初始化失败")
            return
        
        print("手势检测测试开始...")
        print("请在摄像头前出示1-4个手指或挥手")
        print("按 Ctrl+C 停止测试")
        
        # 连续检测多次
        for i in range(5):
            print(f"\n第 {i+1} 次检测:")
            detector.reset_detection_state()
            result = detector.get_finger_number()
            print(f"检测结果: {result}")
            
            if result > 0:
                print(f"成功检测到 {result} 个手指!")
            else:
                print("未检测到有效手势")
            
            time.sleep(1)
        
    except KeyboardInterrupt:
        print("\n测试被用户终止")
    except Exception as e:
        print(f"测试失败: {e}")
    finally:
        detector.close()


if __name__ == "__main__":
    # 选择测试模式
    print("选择测试模式:")
    print("1. 兼容性测试 (使用 FingerNumber 函数)")
    print("2. 类测试 (使用 FingerNumberDetector 类)")
    print("3. 连续检测测试")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        # 兼容性测试
        print("兼容性测试开始...")
        result = FingerNumber()
        print(f"检测结果: {result}")
        
    elif choice == "2":
        # 类测试
        detector = FingerNumberDetector(debug=True)
        try:
            detector.initialize()
            result = detector.get_finger_number()
            print(f"检测结果: {result}")
        finally:
            detector.close()
            
    elif choice == "3":
        # 连续检测测试
        test_finger_detection()
        
    else:
        print("无效选择")