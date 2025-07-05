import cv2
import numpy as np
from ArrowOpenCV import ArrowDetector, ArrowDirection

def debug_arrow_detection():
    """调试箭头检测"""
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 初始化箭头检测器
    detector = ArrowDetector(debug=True)
    
    print("开始调试箭头检测...")
    print("按键说明:")
    print("  'q' - 退出")
    print("  's' - 保存当前帧")
    print("  'c' - 显示/隐藏轮廓")
    print("  'r' - 重置ROI区域")
    
    show_contours = True
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头")
            break
        
        frame_count += 1
        
        # 复制原始帧
        display_frame = frame.copy()
        
        # 检测箭头
        results = detector.detect_arrows(frame)
        
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
                if contour is not None and show_contours:
                    cv2.drawContours(display_frame, [contour], -1, color, 2)
                
                # 添加标签
                label = f"{direction.value}"
                cv2.putText(display_frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 添加调试信息
        info_text = f"Frame: {frame_count} | Detected: {len(results)} | Contours: {'ON' if show_contours else 'OFF'}"
        cv2.putText(display_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 添加ROI区域显示
        height, width = frame.shape[:2]
        roi_x, roi_y = 0, 0
        roi_w, roi_h = width, height
        cv2.rectangle(display_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 255), 1)
        
        # 显示图像
        cv2.imshow('Arrow Detection Debug', display_frame)
        
        # 处理按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 保存当前帧
            timestamp = cv2.getTickCount()
            filename = f"debug_frame_{timestamp}.jpg"
            cv2.imwrite(filename, display_frame)
            print(f"保存图像: {filename}")
        elif key == ord('c'):
            # 切换轮廓显示
            show_contours = not show_contours
            print(f"轮廓显示: {'开启' if show_contours else '关闭'}")
        elif key == ord('r'):
            # 重置检测器
            detector = ArrowDetector(debug=True)
            print("检测器已重置")
    
    # 清理资源
    cap.release()
    cv2.destroyAllWindows()
    print("调试结束")

if __name__ == "__main__":
    debug_arrow_detection()