# coding=utf-8
import time
from Transbot_Lib import Transbot
from ipywidgets import interact, widgets
from typing import List, Optional, Dict, Tuple

class Transbot_ARM:
    # 关节角度范围限制
    JOINT_LIMITS = {
        7: {'min': 0, 'max': 225, 'default': 180},
        8: {'min': 30, 'max': 270, 'default': 180},
        9: {'min': 30, 'max': 180, 'default': 180}
    }
    
    def __init__(self, arm_offset: List[int], debug: bool = False):
        """
        机械臂控制器初始化
        
        参数:
            arm_offset: 机械臂偏移量列表 [offset_7, offset_8, offset_9]
            debug: 是否启用调试模式
        """
        self.object_sequences = {}  # 新增物体-序列映射字典
        self._arm_offset = arm_offset.copy()  # 使用副本避免外部修改
        self.debug = debug
        self.bot = Transbot()
        self.preset_positions = {}  # 预设位置字典
        self._initialize_arm()
        
        if debug:
            print(f"机械臂初始化完成，偏移量: {self._arm_offset}")

    def _initialize_arm(self):
        """初始化机械臂连接"""
        try:
            self.bot.create_receive_threading()
            if self.debug:
                print("机械臂通信线程已启动")
        except Exception as e:
            print(f"机械臂初始化失败: {e}")
            raise

    def register_object_sequence(self, object_type: str, sequence: List[Tuple]):
        """
        注册物体对应的动作序列
        参数:
            object_type: 物体类型标识 (如"cube","ball")
            sequence: 对应的动作序列
        """
        self.object_sequences[object_type] = sequence
        if self.debug:
            print(f"注册物体 '{object_type}' 的动作序列: {len(sequence)}个步骤")

    def execute_for_object(self, object_type: str, delay: float = 1.0):
        """
        执行物体对应的动作序列
        
        参数:
            object_type: 要处理的物体类型
            delay: 默认步骤间隔时间(秒)
        """
        if object_type not in self.object_sequences:
            raise ValueError(f"未注册的物体类型: {object_type}")
        
        sequence = self.object_sequences[object_type]
        print(f"=== 开始处理 {object_type} ===")
        self.execute_movement_sequence(sequence, delay)
        print(f"=== {object_type} 处理完成 ===")
    
    def set_preset_positions(self, positions: Dict[str, List[int]]):
        """设置预设位置字典"""
        self.preset_positions = positions.copy()
        if self.debug:
            print(f"已设置 {len(positions)} 个预设位置")

    def set_joint_angle(self, joint_id: int, angle: Optional[int] = None, 
                       use_offset: bool = True, delay: float = 0.1):
        """
        设置单个关节角度
        
        参数:
            joint_id: 关节ID (7,8,9)
            angle: 目标角度，None表示使用预设偏移量
            use_offset: 是否应用偏移量
            delay: 执行后的延迟时间(秒)
        """
        if joint_id not in self.JOINT_LIMITS:
            raise ValueError(f"关节ID必须是7、8或9，当前输入: {joint_id}")
            
        final_angle = angle if angle is not None else self._arm_offset[joint_id-7]
        if use_offset and angle is None:
            final_angle = self._arm_offset[joint_id-7]
        
        limits = self.JOINT_LIMITS[joint_id]
        final_angle = max(limits['min'], min(limits['max'], final_angle))
        
        self.bot.set_uart_servo_angle(joint_id, final_angle)
        if self.debug:
            print(f"设置关节{joint_id}角度为: {final_angle}°")
        time.sleep(delay)
    
    def set_all_joints(self, 
                      angles: Optional[Dict[int, int]] = None, 
                      use_offset: bool = True,
                      delay: float = 0.1):
        """
        设置多个关节角度
        
        参数:
            angles: 字典格式 {关节ID: 角度}，None表示使用所有预设偏移量
            use_offset: 是否应用偏移量
            delay: 每个关节设置后的延迟时间(秒)
        """
        if angles is None:
            # 使用预设偏移量设置所有关节
            for joint_id in range(7, 10):
                self.set_joint_angle(joint_id, use_offset=use_offset, delay=delay)
        else:
            # 设置指定关节
            for joint_id, angle in angles.items():
                self.set_joint_angle(joint_id, angle, use_offset=use_offset, delay=delay)

    def create_interactive_control(self, joint_id: int):
        """
        创建交互式滑块控制指定关节
        
        参数:
            joint_id: 要控制的关节ID (7, 8, 或9)
        """
        if joint_id not in self.JOINT_LIMITS:
            raise ValueError(f"关节ID必须是7、8或9，当前输入: {joint_id}")
        
        limits = self.JOINT_LIMITS[joint_id]
        
        def update_angle(angle: int):
            """滑块回调函数"""
            self.set_joint_angle(joint_id, angle, use_offset=False)
        
        # 创建交互式滑块
        interact(update_angle, 
                angle=widgets.IntSlider(
                    min=limits['min'],
                    max=limits['max'],
                    value=limits['default'],
                    description=f'关节{joint_id}角度'
                ))

    def move_to_position(self, position_name: str):
        """
        使用预设位置字典调整机械臂
        
        参数:
            position_name: 位置名称
            preset_positions: 预设位置字典
        """
        if position_name not in self.preset_positions:
            raise ValueError(f"未知位置: {position_name}")
            
        angles = {
            7: self.preset_positions[position_name][0],
            8: self.preset_positions[position_name][1],
            9: self.preset_positions[position_name][2]
        }
        self.set_all_joints(angles, use_offset=False, delay=0.3)

    def execute_movement_sequence(self, sequence: List[Tuple], delay: float = 1.0):
        """
        执行二维元组定义的动作序列
        
        参数:
            sequence: 动作序列，每个元组格式为:
                - ('position', 位置名称) 或
                - ('joint', 关节ID, 角度) 或
                - ('delay', 延时秒数)
            delay: 每个动作后的默认延时(秒)
        """
        try:
            for i, action in enumerate(sequence, 1):
                if self.debug:
                    print(f"\n执行动作 {i}/{len(sequence)}: {action}")
                
                if action[0] == 'position':
                    self.move_to_position(action[1])
                elif action[0] == 'joint':
                    self.set_joint_angle(action[1], action[2])
                elif action[0] == 'delay':
                    time.sleep(action[1])
                    continue
                else:
                    raise ValueError(f"未知动作类型: {action[0]}")
                
                time.sleep(delay)
                
        except Exception as e:
            print(f"动作序列执行失败: {e}")
            raise

    def close(self):
        """关闭机械臂连接"""
        try:
            if hasattr(self.bot, 'close'):
                self.bot.close()
            if self.debug:
                print("机械臂连接已关闭")
        except Exception as e:
            print(f"关闭连接时出错: {e}")

    def __del__(self):
        """析构函数，确保资源被释放"""
        self.close()
        if self.debug:
            print('机械臂控制器已销毁')

if __name__ == '__main__':
    
    # 初始化机械臂
    bot = Transbot_ARM([140, 140, 30], debug=True)
    
    # 设置预设位置
    POSITIONS = {
        # 初始安全位置（机械臂完全收起）
        'home': [180, 30, 30],  
        
        # 抓取准备位置（悬停在目标上方）
        'pre_grab': [120, 90, 50],  
        
        # 实际抓取位置（下降到物体高度）
        'grab': [120, 90, 20],  
        
        # 中间过渡位置（避免碰撞）
        'lift': [120, 60, 80],  
        
        # 放置准备位置（悬停在放置点上方）
        'pre_place': [90, 60, 80],  
        
        # 最终放置位置
        'place': [90, 60, 30],  
        
        # 特殊位置（检修/调试）
        'diagnostic': [0, 180, 90]
    }
    bot.set_preset_positions(POSITIONS)
    
    demo_sequence = [
        ('position', 'home'),
        ('position', 'pre_grab'),
        ('position', 'grab'),
        ('position', 'lift'),
        ('position', 'pre_place'),
        ('position', 'place'),
        ('position', 'diagnostic')
    ]
    
    bot.register_object_sequence('demo', demo_sequence)
    try:
        bot.execute_for_object('demo')
    finally:
        del bot
    

