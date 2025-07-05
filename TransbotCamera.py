import time
import cv2
from Transbot_Lib import Transbot

class TransbotCamera:
    """摄像头云台控制器"""
    
    def __init__(self, debug=False):
        """
        初始化摄像头控制器
        :param debug: 是否开启调试模式
        """
        self.bot = None
        self.debug = debug
        
        # 添加摄像头捕获相关
        self.camera_capture = None
        self.camera_id = 0
        self.camera_width = 640
        self.camera_height = 480
        self.camera_fps = 30
        
        # 云台角度配置
        self.HORIZONTAL_SERVO_ID = 1    # 水平舵机ID
        self.VERTICAL_SERVO_ID = 2      # 垂直舵机ID
        
        # 角度范围定义
        self.HORIZONTAL_MIN = 0         # 水平最小角度
        self.HORIZONTAL_MAX = 180       # 水平最大角度
        self.HORIZONTAL_CENTER = 90     # 水平中心角度
        
        self.VERTICAL_MIN = 0           # 垂直最小角度
        self.VERTICAL_MAX = 180         # 垂直最大角度
        self.VERTICAL_CENTER = 90       # 垂直中心角度
        
        # 预设角度位置
        self.PRESET_POSITIONS = {
            'center': (90, 90),           # 中心位置
            'left': (180, 90),            # 左转90度
            'right': (0, 90),             # 右转90度
            'up': (90, 0),                # 向上
            'down': (90, 180),            # 向下
            'left_up': (180, 45),         # 左上
            'left_down': (180, 135),      # 左下
            'right_up': (0, 45),          # 右上
            'right_down': (0, 135),       # 右下
        }
        
        # 当前角度记录
        self.current_horizontal = self.HORIZONTAL_CENTER
        self.current_vertical = self.VERTICAL_CENTER
        
        if self.debug:
            print("摄像头控制器初始化完成")
    
    def initialize_camera(self, camera_id=0, width=640, height=480, fps=30):
        """
        初始化摄像头捕获
        :param camera_id: 摄像头ID
        :param width: 图像宽度
        :param height: 图像高度
        :param fps: 帧率
        :return: 初始化是否成功
        """
        try:
            self.camera_id = camera_id
            self.camera_width = width
            self.camera_height = height
            self.camera_fps = fps
            
            if self.debug:
                print(f"正在初始化摄像头 {camera_id}...")
            
            # 尝试打开摄像头
            self.camera_capture = cv2.VideoCapture(camera_id)
            
            if not self.camera_capture.isOpened():
                if self.debug:
                    print(f"无法打开摄像头 {camera_id}")
                return False
            
            # 设置摄像头参数
            self.camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera_capture.set(cv2.CAP_PROP_FPS, fps)
            self.camera_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # 等待摄像头稳定
            time.sleep(1)
            
            # 清空缓冲区
            for i in range(5):
                ret, frame = self.camera_capture.read()
                if self.debug:
                    print(f"清空缓冲区 {i+1}/5: {'成功' if ret else '失败'}")

            # 测试读取一帧
            ret, frame = self.camera_capture.read()
            if not ret:
                if self.debug:
                    print("摄像头无法读取图像")
                self.camera_capture.release()
                self.camera_capture = None
                return False
            
            # 检查帧质量
            if frame.mean() < 5:
                if self.debug:
                    print(f"警告: 摄像头帧过暗 (平均值: {frame.mean():.2f})")
            
            if self.debug:
                print(f"摄像头初始化成功: {width}x{height}@{fps}fps")
                print(f"实际参数: {int(self.camera_capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.camera_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            
            return True
            
            
        except Exception as e:
            if self.debug:
                print(f"摄像头初始化失败: {e}")
            return False
    
    def get_camera_frame(self):
        """
        获取摄像头帧
        :return: (成功标志, 图像帧)
        """
        if self.camera_capture is None:
            if self.debug:
                print("摄像头未初始化")
            return False, None
        
        try:
            ret, frame = self.camera_capture.read()
            
            if self.debug and ret:
                # 添加调试信息
                if frame is not None:
                    mean_val = frame.mean()
                    if mean_val < 5:
                        print(f"警告: 帧过暗 (平均值: {mean_val:.2f})")
                    elif mean_val > 250:
                        print(f"警告: 帧过亮 (平均值: {mean_val:.2f})")
            
            return ret, frame
            
        except Exception as e:
            if self.debug:
                print(f"读取摄像头帧失败: {e}")
            return False, None
    
    def is_camera_opened(self):
        """
        检查摄像头是否已打开
        :return: 摄像头是否打开
        """
        return self.camera_capture is not None and self.camera_capture.isOpened()
    
    def close_camera(self):
        """关闭摄像头"""
        if self.camera_capture is not None:
            self.camera_capture.release()
            self.camera_capture = None
            if self.debug:
                print("摄像头已关闭")
    
    def initialize(self):
        """
        初始化硬件连接（包括云台和摄像头）
        :return: 初始化是否成功
        """
        try:
            # 初始化云台
            if self.debug:
                print("正在连接Transbot...")
            
            self.bot = Transbot()
            
            if self.debug:
                print("Transbot连接成功")
            
            # 初始化云台到中心位置
            self.reset_to_center()
            
            # 初始化摄像头
            if not self.initialize_camera():
                if self.debug:
                    print("摄像头初始化失败，但云台可以正常使用")
                # 不返回False，允许只使用云台功能
            
            if self.debug:
                print("摄像头控制器初始化完成")
            
            return True
            
        except Exception as e:
            if self.debug:
                print(f"初始化失败: {e}")
            return False
    
    # 保持原有的云台控制功能不变
    def set_horizontal_angle(self, angle):
        """
        设置水平角度
        :param angle: 角度值 (0-180)
        :return: 设置是否成功
        """
        try:
            # 角度范围检查
            if angle < self.HORIZONTAL_MIN or angle > self.HORIZONTAL_MAX:
                if self.debug:
                    print(f"水平角度超出范围: {angle}° (范围: {self.HORIZONTAL_MIN}°-{self.HORIZONTAL_MAX}°)")
                return False
            
            # 设置舵机角度
            self.bot.set_pwm_servo(self.HORIZONTAL_SERVO_ID, angle)
            self.current_horizontal = angle
            
            if self.debug:
                print(f"水平角度设置为: {angle}°")
            
            return True
            
        except Exception as e:
            if self.debug:
                print(f"设置水平角度失败: {e}")
            return False
    
    def set_vertical_angle(self, angle):
        """
        设置垂直角度
        :param angle: 角度值 (0-180)
        :return: 设置是否成功
        """
        try:
            # 角度范围检查
            if angle < self.VERTICAL_MIN or angle > self.VERTICAL_MAX:
                if self.debug:
                    print(f"垂直角度超出范围: {angle}° (范围: {self.VERTICAL_MIN}°-{self.VERTICAL_MAX}°)")
                return False
            
            # 设置舵机角度
            self.bot.set_pwm_servo(self.VERTICAL_SERVO_ID, angle)
            self.current_vertical = angle
            
            if self.debug:
                print(f"垂直角度设置为: {angle}°")
            
            return True
            
        except Exception as e:
            if self.debug:
                print(f"设置垂直角度失败: {e}")
            return False
    
    def set_camera_angle(self, horizontal_angle, vertical_angle, delay=0.5):
        """
        同时设置水平和垂直角度
        :param horizontal_angle: 水平角度
        :param vertical_angle: 垂直角度
        :param delay: 设置间隔时间
        :return: 设置是否成功
        """
        try:
            # 设置水平角度
            success_h = self.set_horizontal_angle(horizontal_angle)
            time.sleep(delay)
            
            # 设置垂直角度
            success_v = self.set_vertical_angle(vertical_angle)
            time.sleep(delay)
            
            if success_h and success_v:
                if self.debug:
                    print(f"摄像头角度设置完成: 水平={horizontal_angle}°, 垂直={vertical_angle}°")
                return True
            else:
                if self.debug:
                    print("摄像头角度设置失败")
                return False
                
        except Exception as e:
            if self.debug:
                print(f"设置摄像头角度失败: {e}")
            return False
    
    def move_to_preset(self, position_name, delay=0.5):
        """
        移动到预设位置
        :param position_name: 预设位置名称
        :param delay: 移动延迟时间
        :return: 移动是否成功
        """
        if position_name not in self.PRESET_POSITIONS:
            if self.debug:
                print(f"未知的预设位置: {position_name}")
                print(f"可用位置: {list(self.PRESET_POSITIONS.keys())}")
            return False
        
        horizontal, vertical = self.PRESET_POSITIONS[position_name]
        
        if self.debug:
            print(f"移动到预设位置: {position_name}")
        
        return self.set_camera_angle(horizontal, vertical, delay)
    
    def reset_to_center(self, delay=0.5):
        """
        重置到中心位置
        :param delay: 重置延迟时间
        :return: 重置是否成功
        """
        if self.debug:
            print("重置摄像头到中心位置")
        
        return self.set_camera_angle(self.HORIZONTAL_CENTER, self.VERTICAL_CENTER, delay)
    
    def turn_left(self, angle=30, delay=0.5):
        """
        左转指定角度
        :param angle: 转动角度 (默认30度)
        :param delay: 转动延迟时间
        :return: 转动是否成功
        """
        new_angle = min(self.current_horizontal + angle, self.HORIZONTAL_MAX)
        
        if self.debug:
            print(f"左转 {angle}°: {self.current_horizontal}° -> {new_angle}°")
        
        return self.set_horizontal_angle(new_angle)
    
    def turn_right(self, angle=30, delay=0.5):
        """
        右转指定角度
        :param angle: 转动角度 (默认30度)
        :param delay: 转动延迟时间
        :return: 转动是否成功
        """
        new_angle = max(self.current_horizontal - angle, self.HORIZONTAL_MIN)
        
        if self.debug:
            print(f"右转 {angle}°: {self.current_horizontal}° -> {new_angle}°")
        
        return self.set_horizontal_angle(new_angle)
    
    def turn_down(self, angle=30, delay=0.5):
        """
        向下转动指定角度
        :param angle: 转动角度 (默认30度)
        :param delay: 转动延迟时间
        :return: 转动是否成功
        """
        new_angle = min(self.current_vertical + angle, self.VERTICAL_MAX)
        
        if self.debug:
            print(f"向下转动 {angle}°: {self.current_vertical}° -> {new_angle}°")
        
        return self.set_vertical_angle(new_angle)
    
    def turn_up(self, angle=30, delay=0.5):
        """
        向上转动指定角度
        :param angle: 转动角度 (默认30度)
        :param delay: 转动延迟时间
        :return: 转动是否成功
        """
        new_angle = max(self.current_vertical - angle, self.VERTICAL_MIN)
        
        if self.debug:
            print(f"向上转动 {angle}°: {self.current_vertical}° -> {new_angle}°")
        
        return self.set_vertical_angle(new_angle)
    
    def get_current_angles(self):
        """
        获取当前角度
        :return: (水平角度, 垂直角度)
        """
        return (self.current_horizontal, self.current_vertical)
    
    def get_preset_positions(self):
        """
        获取所有预设位置
        :return: 预设位置字典
        """
        return self.PRESET_POSITIONS.copy()
    
    def print_status(self):
        """打印当前状态"""
        print(f"\n摄像头控制器状态:")
        print(f"  当前角度: 水平={self.current_horizontal}°, 垂直={self.current_vertical}°")
        print(f"  角度范围: 水平({self.HORIZONTAL_MIN}°-{self.HORIZONTAL_MAX}°), 垂直({self.VERTICAL_MIN}°-{self.VERTICAL_MAX}°)")
        print(f"  预设位置: {list(self.PRESET_POSITIONS.keys())}")
        print(f"  摄像头状态: {'已打开' if self.is_camera_opened() else '未打开'}")
        print()
    
    def close(self):
        """关闭控制器"""
        if self.debug:
            print("关闭摄像头控制器...")
        
        # 关闭摄像头
        self.close_camera()
        
        # 重置到中心位置
        self.reset_to_center()
        
        if self.debug:
            print("摄像头控制器已关闭")

# 添加一个测试摄像头功能的函数
def test_camera_capture():
    """测试摄像头捕获功能"""
    camera = TransbotCamera(debug=True)
    
    try:
        # 初始化
        if not camera.initialize():
            print("初始化失败")
            return
        
        print("摄像头捕获测试开始...")
        print("按 'q' 退出测试")
        
        frame_count = 0
        while True:
            # 获取帧
            ret, frame = camera.get_camera_frame()
            if not ret:
                print("无法获取摄像头帧")
                break
            
            frame_count += 1
            
            # 显示帧
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Camera Test', frame)
            
            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"测试失败: {e}")
    finally:
        camera.close()

if __name__ == "__main__":
    # 选择测试模式
    print("选择测试模式:")
    print("1. 云台控制测试")
    print("2. 摄像头捕获测试")
    
    choice = input("请输入选择 (1/2): ").strip()
    
    if choice == "1":
        # 原有的云台测试
        camera = TransbotCamera(debug=True)
        camera.initialize()
        camera.turn_down()
    elif choice == "2":
        # 新的摄像头测试
        test_camera_capture()
    else:
        print("无效选择")