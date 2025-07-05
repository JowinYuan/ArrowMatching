import time
from Transbot_Lib import Transbot
from typing import List, Tuple
from enum import Enum

class MotionType(Enum):
    """运动类型枚举"""
    FORWARD = "前进"
    LEFT_TURN = "左转"
    RIGHT_TURN = "右转"
    BACKWARD = "后退"
    STOP = "停止"

class TransbotMotion:
    def __init__(self, wheel_distance: float = 0.15, debug: bool = False):
        """
        小车运动控制器
        :param wheel_distance: 轮距(米)
        :param debug: 调试模式
        """
        self.bot = Transbot()
        self.wheel_distance = wheel_distance
        self.debug = debug
        
        # 运动参数
        self.motion_params = {
            'max_velocity': 0.45,       # 最大线速度 (米/秒)
            'max_rotation': 1,          # 最大旋转值 (范围 -2.00 到 2.00)
            'default_velocity': 0.2,    # 默认前进速度
            'rotation_per_180': 1.0,    # 180度对应的旋转值
            'turn_duration': 2.0        # 默认转弯持续时间(秒)
        }
        
        # 校准参数
        self.calibration = {
            'distance_comp': 1.0,       # 距离补偿系数
            'left_turn_comp': 1.02,     # 左转补偿系数
            'right_turn_comp': 1.05,    # 右转补偿系数
            'velocity_comp': 1.0,       # 速度补偿系数
            'left_rotation_comp': 1.02, # 左旋转补偿系数
            'right_rotation_comp': 1.05 # 右旋转补偿系数
        }
        
        if self.debug:
            print("小车运动控制器初始化完成")
    
    def stop(self):
        """停止小车"""
        self.bot.set_car_motion(0.0, 0.0)
        if self.debug:
            print("小车已停止")
    
    def move_forward(self, distance: float = 0.0, velocity: float = None):
        """
        前进指定距离
        :param distance: 前进距离(米)，0表示持续前进
        :param velocity: 前进速度(米/秒)，None表示使用默认速度
        """
        if velocity is None:
            velocity = self.motion_params['default_velocity']
        
        # 限制速度范围
        velocity = max(0, min(self.motion_params['max_velocity'], velocity))
        velocity *= self.calibration['velocity_comp']
        
        # 开始前进
        self.bot.set_car_motion(velocity, 0.0)
        
        if distance > 0:
            # 计算运动时间
            move_time = distance / velocity * self.calibration['distance_comp']
            if self.debug:
                print(f"前进 {distance:.2f} 米，速度 {velocity:.2f} m/s，预计用时 {move_time:.2f} 秒")
            
            time.sleep(move_time)
            self.stop()
        elif self.debug:
            print(f"开始持续前进，速度 {velocity:.2f} m/s")
    
    def move_backward(self, distance: float = 0.0, velocity: float = None):
        """
        后退指定距离
        :param distance: 后退距离(米)，0表示持续后退
        :param velocity: 后退速度(米/秒)，None表示使用默认速度
        """
        if velocity is None:
            velocity = self.motion_params['default_velocity']
        
        # 限制速度范围
        velocity = max(0, min(self.motion_params['max_velocity'], velocity))
        velocity *= self.calibration['velocity_comp']
        
        # 开始后退（负速度）
        self.bot.set_car_motion(-velocity, 0.0)
        
        if distance > 0:
            # 计算运动时间
            move_time = distance / velocity * self.calibration['distance_comp']
            if self.debug:
                print(f"后退 {distance:.2f} 米，速度 {velocity:.2f} m/s，预计用时 {move_time:.2f} 秒")
            
            time.sleep(move_time)
            self.stop()
        elif self.debug:
            print(f"开始持续后退，速度 {velocity:.2f} m/s")
    
    def turn_left(self, angle: float = 90.0, forward_distance: float = 0.0, turn_duration: float = None):
        """
        左转指定角度，然后前进指定距离
        :param angle: 左转角度(度)
        :param forward_distance: 转弯后前进距离(米)
        :param turn_duration: 转弯持续时间(秒)
        """
        if turn_duration is None:
            turn_duration = self.motion_params['turn_duration']
        
        # 计算旋转值：角度 / 180 * rotation_per_180
        rotation_value = (angle / 180) * self.motion_params['rotation_per_180']
        
        # 应用左转旋转补偿
        rotation_value *= self.calibration['left_rotation_comp']
        
        # 限制旋转值范围
        rotation_value = max(-self.motion_params['max_rotation'], 
                           min(self.motion_params['max_rotation'], rotation_value))
        
        # 开始左转（正旋转值为左转）
        self.bot.set_car_motion(0.0, rotation_value)
        
        # 计算转弯时间 - 使用左转时间补偿
        actual_turn_time = turn_duration * (angle / 90.0) * self.calibration['left_turn_comp']
        
        if self.debug:
            print(f"左转 {angle:.1f} 度，旋转值 {rotation_value:.2f}，预计用时 {actual_turn_time:.2f} 秒")
        
        time.sleep(actual_turn_time)
        self.stop()
        
        # 转弯后前进
        if forward_distance > 0:
            time.sleep(0.1)  # 短暂停顿
            self.move_forward(forward_distance)
    
    def turn_right(self, angle: float = 90.0, forward_distance: float = 0.0, turn_duration: float = None):
        """
        右转指定角度，然后前进指定距离
        :param angle: 右转角度(度)
        :param forward_distance: 转弯后前进距离(米)
        :param turn_duration: 转弯持续时间(秒)
        """
        if turn_duration is None:
            turn_duration = self.motion_params['turn_duration']
        
        # 计算旋转值：角度 / 180 * rotation_per_180
        rotation_value = (angle / 180) * self.motion_params['rotation_per_180']
        
        # 应用右转旋转补偿
        rotation_value *= self.calibration['right_rotation_comp']
        
        # 限制旋转值范围
        rotation_value = max(-self.motion_params['max_rotation'], 
                           min(self.motion_params['max_rotation'], rotation_value))
        
        # 开始右转（负旋转值为右转）
        self.bot.set_car_motion(0.0, -rotation_value)
        
        # 计算转弯时间 - 使用右转时间补偿
        actual_turn_time = turn_duration * (angle / 90.0) * self.calibration['right_turn_comp']
        
        if self.debug:
            print(f"右转 {angle:.1f} 度，旋转值 {-rotation_value:.2f}，预计用时 {actual_turn_time:.2f} 秒")
        
        time.sleep(actual_turn_time)
        self.stop()
        
        # 转弯后前进
        if forward_distance > 0:
            time.sleep(0.1)  # 短暂停顿
            self.move_forward(forward_distance)
    
    def set_motion(self, velocity: float, rotation: float):
        """
        直接设置运动参数
        :param velocity: 线速度 (米/秒)，范围 [-0.45, 0.45]
        :param rotation: 旋转值，范围 [-2.00, 2.00]
        """
        # 限制参数范围
        velocity = max(-self.motion_params['max_velocity'], 
                      min(self.motion_params['max_velocity'], velocity))
        rotation = max(-self.motion_params['max_rotation'], 
                      min(self.motion_params['max_rotation'], rotation))
        
        self.bot.set_car_motion(velocity, rotation)
        
        if self.debug:
            print(f"设置运动: 线速度 {velocity:.2f} m/s, 旋转值 {rotation:.2f}")
    
    def turn_by_rotation_value(self, rotation_value: float, duration: float = 2.0):
        """
        按旋转值转弯
        :param rotation_value: 旋转值，正数左转，负数右转，1.0对应180度
        :param duration: 持续时间(秒)
        """
        # 限制旋转值范围
        rotation_value = max(-self.motion_params['max_rotation'], 
                           min(self.motion_params['max_rotation'], rotation_value))
        
        # 根据旋转方向应用不同的补偿
        if rotation_value > 0:  # 左转
            rotation_value *= self.calibration['left_rotation_comp']
        else:  # 右转
            rotation_value *= self.calibration['right_rotation_comp']
        
        # 开始转弯
        self.bot.set_car_motion(0.0, rotation_value)
        
        if self.debug:
            angle_estimate = rotation_value * 180.0 / self.motion_params['rotation_per_180']
            direction = "左转" if rotation_value > 0 else "右转"
            print(f"{direction} 旋转值 {rotation_value:.2f} (约{abs(angle_estimate):.1f}度)，持续 {duration:.2f} 秒")
        
        time.sleep(duration)
        self.stop()
    
    def execute_motion_sequence(self, sequence: List[Tuple[MotionType, float, float]]):
        """
        执行运动序列
        :param sequence: 运动序列，每个元素为 (运动类型, 参数1, 参数2)
                        - FORWARD: (FORWARD, 距离, 速度)
                        - LEFT_TURN: (LEFT_TURN, 角度, 前进距离)
                        - RIGHT_TURN: (RIGHT_TURN, 角度, 前进距离)
                        - BACKWARD: (BACKWARD, 距离, 速度)
                        - STOP: (STOP, 停顿时间, 0)
        """
        if self.debug:
            print(f"开始执行运动序列，共 {len(sequence)} 个动作")
        
        for i, (motion_type, param1, param2) in enumerate(sequence):
            if self.debug:
                print(f"执行动作 {i+1}: {motion_type.value}")
            
            if motion_type == MotionType.FORWARD:
                self.move_forward(param1, param2 if param2 > 0 else None)
            elif motion_type == MotionType.BACKWARD:
                self.move_backward(param1, param2 if param2 > 0 else None)
            elif motion_type == MotionType.LEFT_TURN:
                self.turn_left(param1, param2)
            elif motion_type == MotionType.RIGHT_TURN:
                self.turn_right(param1, param2)
            elif motion_type == MotionType.STOP:
                self.stop()
                if param1 > 0:
                    time.sleep(param1)
            
            # 动作间短暂停顿
            time.sleep(0.1)
        
        if self.debug:
            print("运动序列执行完成")
    
    def calibrate_left_turn(self, test_angle: float = 90.0):
        """
        校准左转参数
        :param test_angle: 测试角度(度)
        """
        print(f"开始左转校准，目标角度: {test_angle} 度")
        print("请观察实际转弯角度，然后输入...")
        
        self.turn_left(test_angle)
        
        try:
            actual_angle = float(input("请输入实际左转角度(度): "))
            if actual_angle > 0:
                compensation = test_angle / actual_angle
                self.calibration['left_rotation_comp'] *= compensation
                print(f"左转旋转补偿系数已更新为: {self.calibration['left_rotation_comp']:.3f}")
            else:
                print("输入无效，校准取消")
        except ValueError:
            print("输入格式错误，校准取消")
    
    def calibrate_right_turn(self, test_angle: float = 90.0):
        """
        校准右转参数
        :param test_angle: 测试角度(度)
        """
        print(f"开始右转校准，目标角度: {test_angle} 度")
        print("请观察实际转弯角度，然后输入...")
        
        self.turn_right(test_angle)
        
        try:
            actual_angle = float(input("请输入实际右转角度(度): "))
            if actual_angle > 0:
                compensation = test_angle / actual_angle
                self.calibration['right_rotation_comp'] *= compensation
                print(f"右转旋转补偿系数已更新为: {self.calibration['right_rotation_comp']:.3f}")
            else:
                print("输入无效，校准取消")
        except ValueError:
            print("输入格式错误，校准取消")
    
    def calibrate_distance(self, test_distance: float = 1.0):
        """
        校准距离参数
        :param test_distance: 测试距离(米)
        """
        print(f"开始距离校准，目标距离: {test_distance} 米")
        print("请手动测量实际行驶距离，然后输入...")
        
        self.move_forward(test_distance)
        
        try:
            actual_distance = float(input("请输入实际行驶距离(米): "))
            if actual_distance > 0:
                compensation = test_distance / actual_distance
                self.calibration['distance_comp'] *= compensation
                print(f"距离补偿系数已更新为: {self.calibration['distance_comp']:.3f}")
            else:
                print("输入无效，校准取消")
        except ValueError:
            print("输入格式错误，校准取消")
    
    def print_calibration_params(self):
        """打印当前校准参数"""
        print("\n=== 当前校准参数 ===")
        print(f"距离补偿系数: {self.calibration['distance_comp']:.3f}")
        print(f"左转时间补偿系数: {self.calibration['left_turn_comp']:.3f}")
        print(f"右转时间补偿系数: {self.calibration['right_turn_comp']:.3f}")
        print(f"左转旋转补偿系数: {self.calibration['left_rotation_comp']:.3f}")
        print(f"右转旋转补偿系数: {self.calibration['right_rotation_comp']:.3f}")
        print(f"速度补偿系数: {self.calibration['velocity_comp']:.3f}")
    
    def set_calibration_params(self, distance_comp: float = None, 
                             left_turn_comp: float = None, right_turn_comp: float = None,
                             left_rotation_comp: float = None, right_rotation_comp: float = None,
                             velocity_comp: float = None):
        """
        手动设置校准参数
        """
        if distance_comp is not None:
            self.calibration['distance_comp'] = distance_comp
        if left_turn_comp is not None:
            self.calibration['left_turn_comp'] = left_turn_comp
        if right_turn_comp is not None:
            self.calibration['right_turn_comp'] = right_turn_comp
        if left_rotation_comp is not None:
            self.calibration['left_rotation_comp'] = left_rotation_comp
        if right_rotation_comp is not None:
            self.calibration['right_rotation_comp'] = right_rotation_comp
        if velocity_comp is not None:
            self.calibration['velocity_comp'] = velocity_comp
        
        if self.debug:
            print("校准参数已更新")
            self.print_calibration_params()
    
    def close(self):
        """关闭连接"""
        self.stop()
        if hasattr(self.bot, '__del__'):
            self.bot.__del__()
        if self.debug:
            print("连接已关闭")

# 使用示例
def main():
    """主函数 - 演示运动控制"""
    # 创建运动控制器
    motion = TransbotMotion(debug=True)
    
    try:
        print("=== 基本运动测试 ===")
        
        # 显示当前校准参数
        motion.print_calibration_params()
        
                
        print("\n 前进0.53")
        motion.move_forward(0.53)

        # 测试右转90度后前进
        print("\n1. 右转90度，然后前进 0.6 米")
        motion.turn_right(90.0, 0.83)
        time.sleep(1)

        # 测试左转90度后前进
        print("\n2. 左转90度，然后前进 0.6 米")
        motion.turn_left(90.0, 1.36)
        time.sleep(1)

        # 前进测试
        print("\n3. 前进 1.06 米")
        motion.move_forward(1.06)
        time.sleep(1)
        
        # 测试直接使用旋转值
        print("\n4. 直接使用旋转值测试")
        motion.turn_by_rotation_value(0.5, 2.0)  # 左转约90度
        time.sleep(1)
        motion.turn_by_rotation_value(-1.0, 2.0) # 右转约180度
        time.sleep(1)
        
        # 后退测试
        print("\n5. 后退0.5米")
        motion.move_backward(0.5)
        time.sleep(1)

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        motion.close()

if __name__ == "__main__":
    main()