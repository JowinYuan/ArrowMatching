import cv2
import numpy as np
import time
from collections import Counter
from ArrowOpenCV import ArrowDetector, ArrowDirection
import os
# 添加环境变量设置，解决Qt显示问题
os.environ['QT_QPA_PLATFORM'] = 'xcb'


class ArrowStatistics:
    """箭头统计类"""
    
    def __init__(self, interval_seconds=5, total_duration=60, debug=True):
        """
        初始化箭头统计
        :param interval_seconds: 统计间隔时间（秒）
        :param total_duration: 总持续时间（秒）
        :param debug: 是否显示调试信息
        """
        self.interval_seconds = interval_seconds
        self.total_duration = total_duration
        self.debug = debug
        
        # 统计数据
        self.arrow_counts = Counter()
        self.detection_history = []
        self.session_results = []
        
        # 时间控制
        self.start_time = None
        self.last_reset_time = None
        
        # 显示控制
        self.show_contours = True
        self.show_stats = True
        
        print(f"箭头统计初始化完成:")
        print(f"  统计间隔: {interval_seconds}秒")
        print(f"  总持续时间: {total_duration}秒")
        print(f"  预计返回次数: {total_duration // interval_seconds}次")
    
    def reset_statistics(self):
        """重置统计数据"""
        self.arrow_counts.clear()
        self.detection_history.clear()
        self.last_reset_time = time.time()
        
        if self.debug:
            print(f"[{time.strftime('%H:%M:%S')}] 统计数据已重置")
    
    def add_detection(self, direction):
        """添加检测结果"""
        if direction != ArrowDirection.UNKNOWN:
            self.arrow_counts[direction] += 1
            self.detection_history.append({
                'direction': direction,
                'timestamp': time.time()
            })
    
    def get_most_common_arrow(self):
        """获取最常见的箭头方向"""
        if not self.arrow_counts:
            return None, 0
        
        most_common = self.arrow_counts.most_common(1)[0]
        return most_common[0], most_common[1]
    
    def should_return_result(self):
        """检查是否应该返回结果"""
        if self.last_reset_time is None:
            return False
        
        elapsed = time.time() - self.last_reset_time
        return elapsed >= self.interval_seconds
    
    def is_finished(self):
        """检查是否完成"""
        if self.start_time is None:
            return False
        
        elapsed = time.time() - self.start_time
        return elapsed >= self.total_duration
    
    def get_elapsed_time(self):
        """获取已用时间"""
        if self.start_time is None:
            return 0
        return time.time() - self.start_time
    
    def get_interval_elapsed_time(self):
        """获取当前间隔已用时间"""
        if self.last_reset_time is None:
            return 0
        return time.time() - self.last_reset_time
    
    def get_statistics_info(self):
        """获取统计信息"""
        total_detections = sum(self.arrow_counts.values())
        return {
            'total_detections': total_detections,
            'left_count': self.arrow_counts.get(ArrowDirection.LEFT, 0),
            'right_count': self.arrow_counts.get(ArrowDirection.RIGHT, 0),
            'straight_count': self.arrow_counts.get(ArrowDirection.STRAIGHT, 0),
            'session_count': len(self.session_results),
            'elapsed_time': self.get_elapsed_time(),
            'interval_elapsed_time': self.get_interval_elapsed_time()
        }

def debug_arrow_statistics():
    """调试箭头统计"""
    # 获取用户输入
    print("=== 箭头统计检测 ===")
    
    try:
        interval = float(input("输入统计间隔时间（秒）[5]: ") or "5")
        duration = float(input("输入总持续时间（秒）[60]: ") or "60")
    except ValueError:
        print("输入无效，使用默认值")
        interval = 5
        duration = 60
    
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 初始化箭头检测器和统计器
    detector = ArrowDetector(debug=True)
    stats = ArrowStatistics(interval_seconds=interval, total_duration=duration, debug=False)
    
    print("\n开始箭头统计检测...")
    print("按键说明:")
    print("  'q' - 退出")
    print("  's' - 保存当前帧")
    print("  'c' - 显示/隐藏轮廓")
    print("  'r' - 手动重置统计")
    print("  'i' - 显示/隐藏统计信息")
    print("  'p' - 暂停/继续")
    
    frame_count = 0
    is_paused = False
    
    # 开始统计
    stats.start_time = time.time()
    stats.last_reset_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头")
            break
        
        frame_count += 1
        
        # 复制原始帧
        display_frame = frame.copy()
        
        # 检查是否暂停
        if not is_paused:
            # 检测箭头
            results = detector.detect_arrows(frame)
            
            # 添加检测结果到统计
            for result in results:
                direction, contour, bbox = result
                stats.add_detection(direction)
            
            # 检查是否应该返回结果
            if stats.should_return_result():
                most_common_direction, count = stats.get_most_common_arrow()
                
                if most_common_direction:
                    result_info = {
                        'direction': most_common_direction,
                        'count': count,
                        'timestamp': time.time(),
                        'session_number': len(stats.session_results) + 1
                    }
                    stats.session_results.append(result_info)
                    
                    print(f"\n[第{result_info['session_number']}次返回] "
                          f"最常见箭头: {most_common_direction.value}, "
                          f"出现次数: {count}")
                    
                    # 显示当前间隔的详细统计
                    info = stats.get_statistics_info()
                    print(f"  本次统计: 左转={info['left_count']}, "
                          f"右转={info['right_count']}, "
                          f"直行={info['straight_count']}, "
                          f"总计={info['total_detections']}")
                else:
                    print(f"\n[第{len(stats.session_results) + 1}次返回] 未检测到箭头")
                
                # 重置统计
                stats.reset_statistics()
            
            # 绘制检测结果
            if results:
                for result in results:
                    direction, contour, bbox = result
                    x, y, w, h = bbox
                    
                    # 根据方向选择颜色
                    if direction == ArrowDirection.LEFT:
                        color = (255, 0, 0)  # 蓝色
                    elif direction == ArrowDirection.RIGHT:
                        color = (0, 0, 255)  # 红色
                    elif direction == ArrowDirection.STRAIGHT:
                        color = (0, 255, 0)  # 绿色
                    else:
                        color = (128, 128, 128)  # 灰色
                    
                    # 绘制边界框
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                    
                    # 绘制轮廓
                    if contour is not None and stats.show_contours:
                        cv2.drawContours(display_frame, [contour], -1, color, 2)
                    
                    # 添加标签
                    label = f"{direction.value}"
                    cv2.putText(display_frame, label, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 添加统计信息显示
        if stats.show_stats:
            add_statistics_overlay(display_frame, stats, frame_count, is_paused)
        
        # 显示图像
        cv2.imshow('Arrow Statistics Detection', display_frame)
        
        # 处理按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 保存当前帧
            timestamp = int(time.time())
            filename = f"stats_frame_{timestamp}.jpg"
            cv2.imwrite(filename, display_frame)
            print(f"保存图像: {filename}")
        elif key == ord('c'):
            # 切换轮廓显示
            stats.show_contours = not stats.show_contours
            print(f"轮廓显示: {'开启' if stats.show_contours else '关闭'}")
        elif key == ord('r'):
            # 手动重置统计
            stats.reset_statistics()
            print("手动重置统计数据")
        elif key == ord('i'):
            # 切换统计信息显示
            stats.show_stats = not stats.show_stats
            print(f"统计信息显示: {'开启' if stats.show_stats else '关闭'}")
        elif key == ord('p'):
            # 暂停/继续
            is_paused = not is_paused
            print(f"检测状态: {'暂停' if is_paused else '继续'}")
        
        # 检查是否完成
        if stats.is_finished():
            print(f"\n=== 统计完成 ===")
            break
    
    # 显示最终结果
    show_final_results(stats)
    
    # 清理资源
    cap.release()
    cv2.destroyAllWindows()
    print("统计结束")

def add_statistics_overlay(image, stats, frame_count, is_paused):
    """添加统计信息覆盖层"""
    try:
        height, width = image.shape[:2]
        
        # 创建半透明背景
        overlay = image.copy()
        alpha = 0.7
        
        # 统计面板位置
        panel_height = 300
        panel_y = height - panel_height
        
        cv2.rectangle(overlay, (0, panel_y), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        # 添加文本信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (255, 255, 255)
        font_thickness = 2
        line_height = 25
        
        y_offset = panel_y + 25
        
        # 状态信息
        status = "PAUSED" if is_paused else "RUNNING"
        status_color = (0, 255, 255) if is_paused else (0, 255, 0)
        cv2.putText(image, f"Status: {status}", (10, y_offset), 
                   font, font_scale, status_color, font_thickness)
        y_offset += line_height
        
        # 时间信息
        total_elapsed = stats.get_elapsed_time()
        interval_elapsed = stats.get_interval_elapsed_time()
        remaining_time = max(0, stats.total_duration - total_elapsed)
        interval_remaining = max(0, stats.interval_seconds - interval_elapsed)
        
        cv2.putText(image, f"Total: {total_elapsed:.1f}s / {stats.total_duration:.1f}s", 
                   (10, y_offset), font, font_scale, font_color, font_thickness)
        y_offset += line_height
        
        cv2.putText(image, f"Interval: {interval_elapsed:.1f}s / {stats.interval_seconds:.1f}s", 
                   (10, y_offset), font, font_scale, font_color, font_thickness)
        y_offset += line_height
        
        cv2.putText(image, f"Remaining: {remaining_time:.1f}s", 
                   (10, y_offset), font, font_scale, (255, 255, 0), font_thickness)
        y_offset += line_height
        
        # 当前统计
        info = stats.get_statistics_info()
        cv2.putText(image, f"Current Stats - Total: {info['total_detections']}", 
                   (10, y_offset), font, font_scale, font_color, font_thickness)
        y_offset += line_height
        
        cv2.putText(image, f"Left: {info['left_count']} | Right: {info['right_count']} | 直行: {info['straight_count']}", 
                   (10, y_offset), font, font_scale, font_color, font_thickness)
        y_offset += line_height
        
        # 返回次数
        cv2.putText(image, f"Returns: {info['session_count']} 次", 
                   (10, y_offset), font, font_scale, (0, 255, 255), font_thickness)
        y_offset += line_height
        
        # 当前最常见方向
        most_common_direction, count = stats.get_most_common_arrow()
        if most_common_direction:
            cv2.putText(image, f"Current Most: {most_common_direction.value} ({count}次)", 
                       (10, y_offset), font, font_scale, (255, 0, 255), font_thickness)
        else:
            cv2.putText(image, "Current Most: None detected", 
                       (10, y_offset), font, font_scale, (128, 128, 128), font_thickness)
        y_offset += line_height
        
        # 进度条
        progress = min(1.0, total_elapsed / stats.total_duration)
        bar_width = width - 20
        bar_height = 20
        bar_x, bar_y = 10, height - 40
        
        # 绘制进度条背景
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # 绘制进度条
        progress_width = int(bar_width * progress)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)
        
        # 进度文本
        cv2.putText(image, f"{progress * 100:.1f}%", 
                   (bar_x + bar_width // 2 - 30, bar_y + 15), 
                   font, 0.5, (255, 255, 255), 1)
        
        # 帧数信息
        cv2.putText(image, f"帧数: {frame_count}", (width - 150, 30), 
                   font, 0.5, (255, 255, 255), 1)
        
    except Exception as e:
        print(f"添加统计覆盖层失败: {e}")

def show_final_results(stats):
    """显示最终结果"""
    print(f"\n{'='*50}")
    print("FINAL STATISTICS RESULTS")
    print(f"{'='*50}")
    
    print(f"Total Duration: {stats.total_duration}秒")
    print(f"Statistics Interval: {stats.interval_seconds}秒")
    print(f"Actual Runtime: {stats.get_elapsed_time():.1f}秒")
    print(f"Return Count: {len(stats.session_results)}")
    
    if stats.session_results:
        print(f"\nEach Return Result:")
        for i, result in enumerate(stats.session_results, 1):
            print(f"  Return #{i}: {result['direction'].value} (appeared {result['count']} times)")
        # 统计各方向被返回的次数
        direction_returns = Counter([result['direction'] for result in stats.session_results])
        print(f"\nDirection Return Statistics:")
        for direction, count in direction_returns.items():
            print(f"  {direction.value}: {count}次")
        
        # 最常被返回的方向
        if direction_returns:
            most_returned = direction_returns.most_common(1)[0]
            print(f"\nMost Returned Direction: {most_returned[0].value} (共{most_returned[1]}次)")
    else:
        print("\nNo return results generated")
    
    print(f"{'='*50}")



def arrow_statistics(show_gui=True):
    """箭头统计检测，返回最常见的箭头方向"""
    print("=== 箭头统计检测 ===")

    interval = 10  # 统计间隔时间
    duration = 10  # 总时间
    
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return "ERROR: 无法打开摄像头"
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cap.set
    
    # 初始化箭头检测器和统计器
    detector = ArrowDetector(debug=False)
    stats = ArrowStatistics(interval_seconds=interval, total_duration=duration, debug=False)
    
    print(f"\n开始箭头统计检测，持续时间: {duration}秒")
    print(f"显示GUI: {'是' if show_gui else '否'}")
    
    frame_count = 0
    last_progress_time = 0
    
    # 开始统计
    stats.start_time = time.time()
    stats.last_reset_time = time.time()
    
    # 全程统计计数器
    total_arrow_counts = Counter()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头")
                break
            
            frame_count += 1
            
            # 检测箭头
            results = detector.detect_arrows(frame)
            
            # 添加检测结果到总统计
            for result in results:
                direction, contour, bbox = result
                if direction != ArrowDirection.UNKNOWN:
                    total_arrow_counts[direction] += 1
                    stats.add_detection(direction)
                    print(f"检测到箭头: {direction.value}")  # 实时显示检测结果
            
            # 如果需要显示GUI
            if show_gui:
                # 复制原始帧
                display_frame = frame.copy()
                
                # 绘制检测结果
                if results:
                    for result in results:
                        direction, contour, bbox = result
                        x, y, w, h = bbox
                        
                        # 根据方向选择颜色
                        if direction == ArrowDirection.LEFT:
                            color = (255, 0, 0)  # 蓝色
                        elif direction == ArrowDirection.RIGHT:
                            color = (0, 0, 255)  # 红色
                        elif direction == ArrowDirection.STRAIGHT:
                            color = (0, 255, 0)  # 绿色
                        else:
                            color = (128, 128, 128)  # 灰色
                        
                        # 绘制边界框
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                        
                        # 绘制轮廓
                        if contour is not None:
                            cv2.drawContours(display_frame, [contour], -1, color, 2)
                        
                        # 添加标签
                        label = f"{direction.value}"
                        cv2.putText(display_frame, label, (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # 添加简单的状态信息
                add_simple_status_overlay(display_frame, stats, total_arrow_counts)
                
                # 显示图像
                try:
                    cv2.imshow('Arrow Statistics Detection', display_frame)
                    # 检查按键
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord('q'):  # ESC 或 'q' 键退出
                        print("用户中断检测")
                        break
                except Exception as e:
                    print(f"GUI显示错误: {e}")
                    show_gui = False  # 禁用GUI继续运行
                    print("切换到无GUI模式")
            
            # 检查是否完成
            if stats.is_finished():
                break
            
            # 改进的进度显示
            current_time = time.time()
            if current_time - last_progress_time >= 2.0:  # 每2秒显示一次
                elapsed = stats.get_elapsed_time()
                progress = (elapsed / duration) * 100
                print(f"检测进度: {elapsed:.1f}/{duration}秒 ({progress:.1f}%)")
                print(f"当前统计: 左转={total_arrow_counts.get(ArrowDirection.LEFT, 0)}, "
                      f"右转={total_arrow_counts.get(ArrowDirection.RIGHT, 0)}, "
                      f"直行={total_arrow_counts.get(ArrowDirection.STRAIGHT, 0)}")
                last_progress_time = current_time
    
    except KeyboardInterrupt:
        print("\n用户中断检测")
    except Exception as e:
        print(f"检测过程中发生错误: {e}")
    finally:
        # 清理资源
        cap.release()
        if show_gui:
            cv2.destroyAllWindows()
    
    # 返回最常见的箭头方向
    if total_arrow_counts:
        most_common_direction, count = total_arrow_counts.most_common(1)[0]
        result = most_common_direction.value
        print(f"\n统计完成，最常见箭头方向: {result} (出现{count}次)")
        
        # 显示详细统计
        print(f"详细统计:")
        for direction, count in total_arrow_counts.items():
            print(f"  {direction.value}: {count}次")
        
        return result
    else:
        print("统计完成，未检测到任何箭头")
        return "未检测到箭头"

def add_simple_status_overlay(image, stats, total_counts):
    """添加简单的状态信息覆盖层"""
    try:
        height, width = image.shape[:2]
        
        # 创建半透明背景
        overlay = image.copy()
        alpha = 0.8
        
        # 状态面板
        panel_height = 80
        panel_y = height - panel_height
        
        cv2.rectangle(overlay, (0, panel_y), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        # 文本设置
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (255, 255, 255)
        font_thickness = 2
        line_height = 25
        
        y_offset = panel_y + 25
        
        # 时间信息
        elapsed = stats.get_elapsed_time()
        progress = (elapsed / stats.total_duration) * 100
        
        cv2.putText(image, f"Time: {elapsed:.1f}s/{stats.total_duration:.1f}s ({progress:.1f}%)", 
                   (10, y_offset), font, font_scale, font_color, font_thickness)
        y_offset += line_height
        
        # 统计信息
        left_count = total_counts.get(ArrowDirection.LEFT, 0)
        right_count = total_counts.get(ArrowDirection.RIGHT, 0)
        straight_count = total_counts.get(ArrowDirection.STRAIGHT, 0)
        total = sum(total_counts.values())
        
        cv2.putText(image, f"Detected: L={left_count} R={right_count} S={straight_count} Total={total}", 
                   (10, y_offset), font, font_scale, (0, 255, 255), font_thickness)
        y_offset += line_height
        
        # 当前最多的方向
        if total_counts:
            most_common = total_counts.most_common(1)[0]
            cv2.putText(image, f"Most Common: {most_common[0].value} ({most_common[1]} times)", 
                       (10, y_offset), font, font_scale, (255, 0, 255), font_thickness)
        
    except Exception as e:
        print(f"添加状态覆盖层失败: {e}")

# 添加无GUI版本
def arrow_statistics_no_gui():
    """无GUI版本的箭头统计"""
    return arrow_statistics(show_gui=False)

# 修改主函数
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'nogui':
            result = arrow_statistics_no_gui()
        elif sys.argv[1] == 'debug':
            debug_arrow_statistics()
        else:
            result = arrow_statistics()
    else:
        result = arrow_statistics()
    
    print(f"最终结果: {result}")