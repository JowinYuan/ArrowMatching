import time
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
    
    def initialize(self):
        """
        初始化硬件连接
        :return: 初始化是否成功
        """
        try:
            if self.debug:
                print("正在连接Transbot...")
            
            self.bot = Transbot()
            
            if self.debug:
                print("Transbot连接成功")
            
            # 初始化云台到中心位置
            self.reset_to_center()
            
            if self.debug:
                print("摄像头控制器初始化完成")
            
            return True
            
        except Exception as e:
            if self.debug:
                print(f"初始化失败: {e}")
            return False
    
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
        :param angle: 转动角度 (默认90度)
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
        :param angle: 转动角度 (默认90度)
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
        new_angle = max(self.current_vertical - angle, self.VERTICAL_MIN)
        
        if self.debug:
            print(f"向上转动 {angle}°: {self.current_vertical}° -> {new_angle}°")
        
        return self.set_vertical_angle(new_angle)
    
    def turn_up(self, angle=30, delay=0.5):
        """
        向上转动指定角度
        :param angle: 转动角度 (默认30度)
        :param delay: 转动延迟时间
        :return: 转动是否成功
        """
        new_angle = min(self.current_vertical + angle, self.VERTICAL_MAX)
        
        if self.debug:
            print(f"向下转动 {angle}°: {self.current_vertical}° -> {new_angle}°")
        
        return self.set_vertical_angle(new_angle)
    
    def scan_horizontal(self, start_angle=0, end_angle=180, step=30, delay=1.0):
        """
        水平扫描
        :param start_angle: 起始角度
        :param end_angle: 结束角度
        :param step: 步长
        :param delay: 每步延迟时间
        :return: 扫描是否成功
        """
        if self.debug:
            print(f"开始水平扫描: {start_angle}° -> {end_angle}°, 步长: {step}°")
        
        try:
            angles = list(range(start_angle, end_angle + 1, step))
            
            for angle in angles:
                if not self.set_horizontal_angle(angle):
                    return False
                time.sleep(delay)
            
            if self.debug:
                print("水平扫描完成")
            
            return True
            
        except Exception as e:
            if self.debug:
                print(f"水平扫描失败: {e}")
            return False
    
    def scan_vertical(self, start_angle=0, end_angle=180, step=30, delay=1.0):
        """
        垂直扫描
        :param start_angle: 起始角度
        :param end_angle: 结束角度
        :param step: 步长
        :param delay: 每步延迟时间
        :return: 扫描是否成功
        """
        if self.debug:
            print(f"开始垂直扫描: {start_angle}° -> {end_angle}°, 步长: {step}°")
        
        try:
            angles = list(range(start_angle, end_angle + 1, step))
            
            for angle in angles:
                if not self.set_vertical_angle(angle):
                    return False
                time.sleep(delay)
            
            if self.debug:
                print("垂直扫描完成")
            
            return True
            
        except Exception as e:
            if self.debug:
                print(f"垂直扫描失败: {e}")
            return False
    
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
    
    def add_preset_position(self, name, horizontal_angle, vertical_angle):
        """
        添加新的预设位置
        :param name: 位置名称
        :param horizontal_angle: 水平角度
        :param vertical_angle: 垂直角度
        :return: 添加是否成功
        """
        try:
            # 检查角度范围
            if (self.HORIZONTAL_MIN <= horizontal_angle <= self.HORIZONTAL_MAX and
                self.VERTICAL_MIN <= vertical_angle <= self.VERTICAL_MAX):
                
                self.PRESET_POSITIONS[name] = (horizontal_angle, vertical_angle)
                
                if self.debug:
                    print(f"添加预设位置: {name} -> ({horizontal_angle}°, {vertical_angle}°)")
                
                return True
            else:
                if self.debug:
                    print(f"角度超出范围: ({horizontal_angle}°, {vertical_angle}°)")
                return False
                
        except Exception as e:
            if self.debug:
                print(f"添加预设位置失败: {e}")
            return False
    
    def print_status(self):
        """打印当前状态"""
        print(f"\n摄像头控制器状态:")
        print(f"  当前角度: 水平={self.current_horizontal}°, 垂直={self.current_vertical}°")
        print(f"  角度范围: 水平({self.HORIZONTAL_MIN}°-{self.HORIZONTAL_MAX}°), 垂直({self.VERTICAL_MIN}°-{self.VERTICAL_MAX}°)")
        print(f"  预设位置: {list(self.PRESET_POSITIONS.keys())}")
        print()
    
    def demo_sequence(self):
        """演示序列"""
        if self.debug:
            print("开始演示序列...")
        
        # 演示所有预设位置
        for position_name in self.PRESET_POSITIONS:
            if self.debug:
                print(f"移动到: {position_name}")
            
            self.move_to_preset(position_name, delay=1.0)
            time.sleep(2.0)  # 停留2秒
        
        # 回到中心位置
        self.reset_to_center()
        
        if self.debug:
            print("演示序列完成")
    
    def close(self):
        """关闭控制器"""
        if self.debug:
            print("关闭摄像头控制器...")
        
        # 重置到中心位置
        self.reset_to_center()
        
        if self.debug:
            print("摄像头控制器已关闭")

if __name__ == "__main__":
    camera = TransbotCamera(debug = True)
    camera.initialize()
    camera.turn_down()
    