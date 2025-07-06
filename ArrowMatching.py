import cv2
import numpy as np
import time
from collections import Counter
from util.ArrowOpenCV import ArrowDetector, ArrowDirection
import os
from util.TransbotCamera import TransbotCamera

# 添加环境变量设置，解决Qt显示问题
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# opencv实现箭头检测

import cv2
import numpy as np
from enum import Enum
from typing import Tuple, List, Dict


class ArrowDirection(Enum):
    """箭头方向枚举"""
    LEFT = "Left"
    RIGHT = "Right"
    STRAIGHT = "Straight"
    UNKNOWN = "Unknown"


class ArrowDetector:
    def __init__(self, debug: bool = False):
        """
        箭头检测器 - 简化版本
        """
        self.debug = debug

        # 红色检测参数保持不变
        self.red_lower_hsv1 = np.array([0, 50, 50])
        self.red_upper_hsv1 = np.array([10, 255, 255])
        self.red_lower_hsv2 = np.array([170, 50, 50])
        self.red_upper_hsv2 = np.array([180, 255, 255])

        self.red_lower_bgr = np.array([0, 0, 150])
        self.red_upper_bgr = np.array([100, 100, 255])

        self.gray_threshold = 25

        # 轮廓检测参数 - 放宽条件
        self.min_contour_area = 200  # 适当提高最小面积
        self.max_contour_area = 50000  # 提高最大面积

        # ROI参数
        self.roi_ratio = 1  # 使用全图

        # 模板匹配参数 - 降低要求
        self.template_match_threshold = 0.05  # 降低阈值
        self.template_sizes = [(150, 150), (170, 170), (180, 180), (190, 190), (200, 200), (220, 220), (240, 240),
                               (260, 260), (270, 270), (280, 280), (290, 290), (300, 300)]  # 简化尺寸

        # 模板形状参数
        self.template_thickness = 8
        self.arrow_head_ratio = 0.35
        self.stem_length_ratio = 0.7

        # 特征匹配参数 - 简化
        self.orb_nfeatures = 150
        self.orb_scale_factor = 1.2
        self.orb_nlevels = 6

        self.feature_match_ratio = 0.7
        self.min_match_count = 4

        # 结果融合参数
        self.template_weight = 0.5  # 降低模板权重
        self.feature_weight = 0.4  # 降低特征权重
        self.min_confidence = 0.15  # 降低最小置信度

        # 【优化】ORB参数 - 更适合箭头检测
        self.orb = cv2.ORB_create(
            nfeatures=self.orb_nfeatures,
            scaleFactor=self.orb_scale_factor,
            nlevels=self.orb_nlevels,
            edgeThreshold=31,  # 边缘阈值，避免边缘噪声
            firstLevel=0,  # 第一层级别
            WTA_K=2,  # 用于产生描述子的随机点数
            scoreType=cv2.ORB_HARRIS_SCORE,  # 使用Harris角点检测
            patchSize=31,  # 描述子使用的patch大小
            fastThreshold=20  # FAST角点检测阈值
        )

        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # 创建模板
        self.templates = {}
        self.template_features = {}
        self._create_arrow_templates()

    def extract_roi(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        提取感兴趣区域（图像中心）
        :param image: 输入图像
        :return: ROI图像和ROI坐标(x, y, w, h)
        """
        h, w = image.shape[:2]

        # 计算ROI区域
        roi_w = int(w * self.roi_ratio)
        roi_h = int(h * self.roi_ratio)
        roi_x = (w - roi_w) // 2
        roi_y = (h - roi_h) // 2

        # 提取ROI
        roi = image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        return roi, (roi_x, roi_y, roi_w, roi_h)

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        图像预处理 - 多种方法结合检测红色
        :param image: 输入图像
        :return: 预处理后的二值图像和ROI坐标
        """
        # 提取ROI
        roi, roi_coords = self.extract_roi(image)

        # 方法1: HSV色彩空间红色检测
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 红色有两个HSV范围，需要分别检测然后合并
        mask_hsv1 = cv2.inRange(hsv, self.red_lower_hsv1, self.red_upper_hsv1)
        mask_hsv2 = cv2.inRange(hsv, self.red_lower_hsv2, self.red_upper_hsv2)
        mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)

        # 方法2: BGR色彩空间红色检测
        mask_bgr = cv2.inRange(roi, self.red_lower_bgr, self.red_upper_bgr)

        # 方法3: 基于红色通道的检测
        # 提取红色通道，红色在红色通道中较亮
        b, g, r = cv2.split(roi)
        # 红色通道减去绿色和蓝色通道，突出红色
        red_enhanced = cv2.subtract(r, cv2.addWeighted(g, 0.5, b, 0.5, 0))
        _, mask_red_channel = cv2.threshold(red_enhanced, 80, 255, cv2.THRESH_BINARY)

        # 方法4: 灰度阈值法作为辅助
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask_gray = cv2.threshold(gray, self.gray_threshold, 255, cv2.THRESH_BINARY)

        # 组合多种方法的结果
        combined_mask = cv2.bitwise_or(mask_hsv, mask_bgr)
        combined_mask = cv2.bitwise_or(combined_mask, mask_red_channel)

        # 与灰度掩码进行交集，去除过暗的区域
        combined_mask = cv2.bitwise_and(combined_mask, mask_gray)

        # 形态学操作去除噪声
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # 进一步去噪
        kernel_large = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_large)

        if self.debug:
            print(f"ROI区域: {roi_coords}")

        return combined_mask, roi_coords

    def find_arrow_contours(self, binary_image: np.ndarray) -> List[np.ndarray]:
        """
        查找箭头轮廓
        :param binary_image: 二值图像
        :return: 符合条件的轮廓列表
        """
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 过滤轮廓
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area < area < self.max_contour_area:
                # 额外的形状过滤
                # 计算轮廓的边界框
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h

                # 过滤掉太细长或太扁的形状
                if 0.3 < aspect_ratio < 3.0:
                    valid_contours.append(contour)

        return valid_contours

    def classify_arrow_direction(self, contour: np.ndarray) -> ArrowDirection:
        """
        简化的箭头方向分类 - 先基于几何特征初步判断，再用模板匹配确认
        """
        # 获取轮廓的基本几何特征
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if self.debug:
            print(f"\n=== 分析轮廓 ===")
            print(f"轮廓边界框: {x}, {y}, {w}, {h}")
            print(f"轮廓面积: {area}")

        # 步骤1-4: 基本筛选保持不变
        if area < self.min_contour_area:
            if self.debug:
                print(f"轮廓面积太小: {area} < {self.min_contour_area}")
            return ArrowDirection.UNKNOWN

        aspect_ratio = w / h
        if aspect_ratio < 0.2 or aspect_ratio > 5.0:
            if self.debug:
                print(f"轮廓宽高比不合理: {aspect_ratio}")
            return ArrowDirection.UNKNOWN

        perimeter = cv2.arcLength(contour, True)
        if perimeter < 50:
            if self.debug:
                print(f"轮廓周长太小: {perimeter}")
            return ArrowDirection.UNKNOWN

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if solidity < 0.2:
                if self.debug:
                    print(f"轮廓凸性太低: {solidity}")
                return ArrowDirection.UNKNOWN

        if self.debug:
            print(f"轮廓通过初步筛选 - 面积: {area}, 宽高比: {aspect_ratio:.2f}, 周长: {perimeter:.1f}")

        # 【关键修改】步骤5: 创建轮廓掩码用于模板匹配和特征匹配
        padding = 10
        contour_mask = np.zeros((h + 2 * padding, w + 2 * padding), dtype=np.uint8)

        # 调整轮廓坐标
        adjusted_contour = contour.copy()
        adjusted_contour[:, :, 0] -= x - padding
        adjusted_contour[:, :, 1] -= y - padding
        cv2.fillPoly(contour_mask, [adjusted_contour], 255)

        # 【关键修改】步骤6: 使用模板匹配和特征匹配
        template_results = self._template_matching_for_contour(contour_mask)
        feature_results = self._feature_matching_for_contour(contour_mask)

        # 【关键修改】步骤7: 结果融合
        final_direction = self._combine_results(template_results, feature_results)

        if self.debug:
            print(f"模板匹配结果: {template_results}")
            print(f"特征匹配结果: {feature_results}")
            print(f"最终方向: {final_direction.value}")

        return final_direction

    def _template_matching_for_contour(self, contour_mask: np.ndarray) -> Dict[ArrowDirection, float]:
        """
        对单个轮廓进行模板匹配
        """
        template_scores = {}

        for template_name, template in self.templates.items():
            best_score = 0

            # 多尺度模板匹配
            for target_size in [min(contour_mask.shape), max(contour_mask.shape)]:
                if target_size > 0:
                    scale = target_size / max(template.shape)
                    if scale > 0.5 and scale < 2.0:  # 合理的缩放范围
                        new_size = (int(template.shape[1] * scale), int(template.shape[0] * scale))
                        if new_size[0] > 0 and new_size[1] > 0:
                            try:
                                scaled_template = cv2.resize(template, new_size)

                                # 确保模板不大于轮廓掩码
                                if (scaled_template.shape[0] <= contour_mask.shape[0] and
                                        scaled_template.shape[1] <= contour_mask.shape[1]):

                                    # 模板匹配
                                    match_result = cv2.matchTemplate(contour_mask, scaled_template,
                                                                     cv2.TM_CCOEFF_NORMED)
                                    _, max_val, _, _ = cv2.minMaxLoc(match_result)

                                    if max_val > best_score:
                                        best_score = max_val

                                    if self.debug:
                                        print(f"  {template_name} (尺寸{new_size}): {max_val:.3f}")
                            except Exception as e:
                                if self.debug:
                                    print(f"  {template_name} 缩放失败: {e}")

            if best_score >= self.template_match_threshold:
                direction = self._template_name_to_direction(template_name)
                if direction not in template_scores or best_score > template_scores[direction]:
                    template_scores[direction] = best_score

        return template_scores

    def _combine_results(self, template_results: Dict[ArrowDirection, float],
                         feature_results: Dict[ArrowDirection, float]) -> ArrowDirection:
        """
        综合模板匹配和特征匹配的结果
        """
        final_scores = {}

        # 获取所有可能的方向
        all_directions = set(template_results.keys()) | set(feature_results.keys())

        for direction in all_directions:
            template_score = template_results.get(direction, 0)
            feature_score = feature_results.get(direction, 0)

            # 加权融合
            combined_score = (template_score * self.template_weight +
                              feature_score * self.feature_weight)

            if combined_score >= self.min_confidence:
                final_scores[direction] = combined_score

                if self.debug:
                    print(
                        f"  {direction.value}: 模板={template_score:.3f}, 特征={feature_score:.3f}, 综合={combined_score:.3f}")

        # 选择得分最高的方向
        if final_scores:
            best_direction = max(final_scores, key=final_scores.get)
            if self.debug:
                print(f"  最佳方向: {best_direction.value} (得分: {final_scores[best_direction]:.3f})")
            return best_direction
        else:
            if self.debug:
                print("  所有方向得分都低于最小置信度")
            return ArrowDirection.UNKNOWN

    def _feature_matching_for_contour(self, contour_mask: np.ndarray) -> Dict[ArrowDirection, float]:
        """
        优化的特征匹配 - 解决调头箭头误匹配问题
        """
        # 提取轮廓特征
        keypoints, descriptors = self.orb.detectAndCompute(contour_mask, None)

        if descriptors is None or len(descriptors) < 5:
            if self.debug:
                print("特征点不足，跳过特征匹配")
            return {}

        if self.debug:
            print(f"提取到 {len(descriptors)} 个特征点")

        feature_scores = {}

        for template_name, template_data in self.template_features.items():
            # 【修改】检查模板特征数据结构
            if template_data is None:
                continue

            if isinstance(template_data, dict):
                template_descriptors = template_data.get('descriptors')
            else:
                template_descriptors = template_data

            if template_descriptors is None or len(template_descriptors) < 5:
                continue

            try:
                # 使用KNN匹配替代暴力匹配
                matches = self.bf.knnMatch(descriptors, template_descriptors, k=2)

                # 改进的Lowe's ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        # 更严格的ratio test，避免错误匹配
                        if m.distance < 0.7 * n.distance:  # 从0.8改为0.7
                            good_matches.append(m)

                if len(good_matches) >= self.min_match_count:
                    # 【关键优化】计算匹配质量时考虑箭头类型
                    match_score = self._calculate_improved_match_score(
                        good_matches, template_name, contour_mask.shape
                    )

                    direction = self._template_name_to_direction(template_name)
                    if direction not in feature_scores or match_score > feature_scores[direction]:
                        feature_scores[direction] = match_score

                    if self.debug:
                        print(f"  {template_name}: {len(good_matches)} 个好匹配, 得分: {match_score:.3f}")
                else:
                    if self.debug:
                        print(f"  {template_name}: 匹配点不足 ({len(good_matches)} < {self.min_match_count})")

            except Exception as e:
                if self.debug:
                    print(f"  {template_name}: 匹配失败 - {e}")

        return feature_scores

    def _calculate_improved_match_score(self, good_matches: List, template_name: str, contour_shape: tuple) -> float:
        """
        改进的匹配得分计算 - 考虑箭头类型特征
        """
        if not good_matches:
            return 0.0

        # 基础得分：基于匹配距离
        distances = [m.distance for m in good_matches]
        avg_distance = np.mean(distances)
        base_score = max(0, 1 - avg_distance / 100)

        # 【关键优化1】匹配点数量惩罚 - 避免复杂形状误匹配
        num_matches = len(good_matches)

        # 初始化所有惩罚变量
        complexity_penalty = 0.0
        distribution_penalty = 0.0
        shape_consistency = 0.0

        # 【关键修改】针对不同箭头类型的复杂度惩罚优化
        if 'straight' in template_name.lower():
            # 直行箭头特殊处理 - 削弱得分
            straight_penalty = 0.15

            if num_matches > 15:
                complexity_penalty = 0.25
            elif num_matches > 8:
                complexity_penalty = 0.15
            elif num_matches < 5:
                complexity_penalty = 0.2
            else:
                complexity_penalty = 0.1

            complexity_penalty += straight_penalty

        elif 'left' in template_name.lower():
            # 【新增】左转箭头特殊处理 - 增强区分度
            if num_matches > 30:
                complexity_penalty = 0.1  # 比右转更宽松
            elif num_matches > 20:
                complexity_penalty = 0.05  # 比右转更宽松
            elif num_matches < 4:
                complexity_penalty = 0.15  # 低匹配数时的惩罚
            else:
                complexity_penalty = 0.0  # 正常情况无惩罚

        elif 'right' in template_name.lower():
            # 【修改】右转箭头处理 - 相对严格一些
            if num_matches > 25:  # 降低阈值，从30降到25
                complexity_penalty = 0.15  # 保持原有惩罚
            elif num_matches > 15:  # 降低阈值，从20降到15
                complexity_penalty = 0.08  # 增加惩罚，从0.05增加到0.08
            elif num_matches < 4:
                complexity_penalty = 0.2  # 增加低匹配数惩罚
            else:
                complexity_penalty = 0.05  # 即使正常情况也有轻微惩罚

        # 【关键优化2】匹配点分布分析 - 对直行箭头更严格
        if num_matches >= 4:
            # 提取匹配点坐标
            query_pts = np.float32([good_matches[i].queryIdx for i in range(num_matches)])

            # 计算匹配点分布的标准差
            if len(query_pts) > 1:
                std_dev = np.std(query_pts)

                # 【针对直行箭头】更严格的分布检查
                if 'straight' in template_name.lower():
                    if std_dev < 3:  # 提高阈值，从2提高到3
                        distribution_penalty = 0.25  # 增加惩罚，从0.2增加到0.25
                    elif std_dev < 5:  # 新增中等惩罚
                        distribution_penalty = 0.15
                    else:
                        distribution_penalty = 0.0
                elif 'left' in template_name.lower():
                    # 【新增】左转箭头分布检查 - 更宽松
                    if std_dev < 1.5:  # 比右转更严格的阈值
                        distribution_penalty = 0.15
                    elif std_dev < 3:
                        distribution_penalty = 0.08
                    else:
                        distribution_penalty = 0.0
                elif 'right' in template_name.lower():
                    # 【修改】右转箭头分布检查 - 相对严格
                    if std_dev < 2:  # 保持原有阈值
                        distribution_penalty = 0.2
                    elif std_dev < 4:
                        distribution_penalty = 0.1
                    else:
                        distribution_penalty = 0.0
                        # 其他箭头保持原有逻辑
                    if std_dev < 2:
                        distribution_penalty = 0.2
                    else:
                        distribution_penalty = 0.0
            else:
                distribution_penalty = 0.0
        else:
            distribution_penalty = 0.0

        # 【关键优化3】形状一致性检查 - 直行箭头更严格
        h, w = contour_shape
        aspect_ratio = w / h if h > 0 else 1.0

        if 'straight' in template_name.lower():
            # 【修改】直行箭头更严格的形状一致性检查
            if aspect_ratio < 0.6:  # 提高要求，从0.8提高到0.6
                shape_consistency = 0.05  # 减少奖励，从0.1减少到0.05
            elif aspect_ratio < 0.9:
                shape_consistency = -0.05  # 新增轻微惩罚
            else:
                shape_consistency = -0.1  # 宽高比不符合直行箭头特征时惩罚
        elif 'left' in template_name.lower():
            # 【新增】左转箭头形状一致性检查 - 给予更多奖励
            if aspect_ratio > 1.4:  # 提高阈值，从1.2提高到1.4
                shape_consistency = 0.15  # 增加奖励，从0.1增加到0.15
            elif aspect_ratio > 1.0:
                shape_consistency = 0.08  # 新增中等奖励
            elif aspect_ratio < 0.8:
                shape_consistency = -0.1  # 太瘦长时惩罚
            else:
                shape_consistency = 0.0
        elif 'right' in template_name.lower():
            # 【修改】右转箭头形状一致性检查 - 相对严格
            if aspect_ratio > 1.3:  # 提高阈值，从1.2提高到1.3
                shape_consistency = 0.12  # 适度奖励，从0.1增加到0.12
            elif aspect_ratio > 1.0:
                shape_consistency = 0.05  # 新增中等奖励
            elif aspect_ratio < 0.7:  # 提高阈值，从0.8提高到0.7
                shape_consistency = -0.12  # 增加惩罚
            else:
                shape_consistency = 0.0

        # 【新增】直行箭头额外的距离惩罚
        straight_distance_penalty = 0.0
        if 'straight' in template_name.lower():
            # 如果平均距离过大，额外惩罚直行箭头
            if avg_distance > 30:
                straight_distance_penalty = 0.1
            elif avg_distance > 20:
                straight_distance_penalty = 0.05

            # 【新增】左转和右转的额外区分机制
        direction_bias = 0.0
        if 'left' in template_name.lower():
            # 给左转箭头一个小的额外优势
            direction_bias = 0.02
        elif 'right' in template_name.lower():
            # 给右转箭头一个小的额外惩罚
            direction_bias = -0.02

        # 综合得分
        final_score = (base_score - complexity_penalty - distribution_penalty +
                       shape_consistency - straight_distance_penalty + direction_bias)
        final_score = max(0.0, min(1.0, final_score))

        if self.debug:
            print(f"    详细得分 - 基础: {base_score:.3f}, 复杂度惩罚: {complexity_penalty:.3f}, "
                  f"分布惩罚: {distribution_penalty:.3f}, 形状一致性: {shape_consistency:.3f}, "
                  f"距离惩罚: {straight_distance_penalty:.3f}, 最终: {final_score:.3f}")

        return final_score

    def detect_arrows(self, image: np.ndarray) -> List[Tuple[ArrowDirection, np.ndarray, Tuple[int, int, int, int]]]:
        """
        检测图像中的箭头 - 简化版本
        """
        # 预处理图像
        binary_image, roi_coords = self.preprocess_image(image)

        # 查找轮廓
        contours = self.find_arrow_contours(binary_image)

        if self.debug:
            print(f"找到 {len(contours)} 个候选轮廓")

        results = []
        for i, contour in enumerate(contours):
            if self.debug:
                print(f"\n=== 分析轮廓 {i + 1}/{len(contours)} ===")

            # 使用简化的方向分类
            direction = self.classify_arrow_direction(contour)

            if direction != ArrowDirection.UNKNOWN:
                # 获取边界框（转换回原图坐标）
                x, y, w, h = cv2.boundingRect(contour)
                x += roi_coords[0]
                y += roi_coords[1]

                # 转换轮廓坐标
                contour_original = contour.copy()
                contour_original[:, :, 0] += roi_coords[0]
                contour_original[:, :, 1] += roi_coords[1]

                results.append((direction, contour_original, (x, y, w, h)))

                if self.debug:
                    print(f"✓ 检测到箭头: {direction.value}")
            else:
                if self.debug:
                    print(f"✗ 轮廓被过滤")

        return results

    def _create_arrow_templates(self):
        """
        创建箭头模板
        """
        if self.debug:
            print("开始创建箭头模板...")

        for size in self.template_sizes:
            w, h = size

            # 计算关键尺寸
            center_x, center_y = w // 2, h // 2
            head_size = int(w * self.arrow_head_ratio)
            stem_length = int(h * self.stem_length_ratio)

            # 1. 创建直行箭头模板 (↑)
            straight_template = self._create_straight_template(w, h, center_x, center_y, head_size, stem_length)

            # 2. 创建左转箭头模板 (L型，头部向左)
            left_template = self._create_left_template(w, h, center_x, center_y, head_size)

            # 3. 创建右转箭头模板 (L型，头部向右)
            right_template = self._create_right_template(w, h, center_x, center_y, head_size)

            # 存储模板
            self.templates[f"straight_{w}x{h}"] = straight_template
            self.templates[f"left_{w}x{h}"] = left_template
            self.templates[f"right_{w}x{h}"] = right_template

            if self.debug:
                print(f"创建模板: {w}x{h} - 线条粗细: {self.template_thickness}")

        # 提取模板特征
        self._extract_template_features()

    def _create_straight_template(self, w: int, h: int, cx: int, cy: int, head_size: int,
                                  stem_length: int) -> np.ndarray:
        """
        创建直行箭头模板
        """
        template = np.zeros((h, w), dtype=np.uint8)

        # 主干：从底部到顶部的垂直线
        stem_start_y = h - 5
        stem_end_y = cy - stem_length // 2
        cv2.line(template, (cx, stem_start_y), (cx, stem_end_y), 255, self.template_thickness)

        # 箭头头部：三角形
        head_top_y = max(2, stem_end_y - head_size)
        head_left_x = cx - head_size // 2
        head_right_x = cx + head_size // 2

        # 绘制三角形箭头头部
        points = np.array([[cx, head_top_y],
                           [head_left_x, stem_end_y],
                           [head_right_x, stem_end_y]], np.int32)
        cv2.fillPoly(template, [points], 511)

        return template

    def _create_left_template(self, w: int, h: int, cx: int, cy: int, head_size: int) -> np.ndarray:
        """
        创建左转箭头模板
        """
        template = np.zeros((h, w), dtype=np.uint8)

        # 垂直段：从底部到中心
        vertical_start_y = h - 5
        vertical_end_y = cy + 3
        cv2.line(template, (cx, vertical_start_y), (cx, vertical_end_y), 255, self.template_thickness)

        # 水平段：从中心到左边
        horizontal_start_x = cx
        horizontal_end_x = 8
        cv2.line(template, (horizontal_start_x, vertical_end_y), (horizontal_end_x, vertical_end_y), 255,
                 self.template_thickness)

        # 箭头头部：指向左边
        head_tip_x = max(2, horizontal_end_x - head_size // 2)
        head_top_y = vertical_end_y - head_size // 2
        head_bottom_y = vertical_end_y + head_size // 2

        # 绘制三角形箭头头部
        points = np.array([[head_tip_x, vertical_end_y],
                           [horizontal_end_x + head_size // 2, head_top_y],
                           [horizontal_end_x + head_size // 2, head_bottom_y]], np.int32)
        cv2.fillPoly(template, [points], 255)

        return template

    def _create_right_template(self, w: int, h: int, cx: int, cy: int, head_size: int) -> np.ndarray:
        """
        创建右转箭头模板
        """
        template = np.zeros((h, w), dtype=np.uint8)

        # 垂直段：从底部到中心
        vertical_start_y = h - 5
        vertical_end_y = cy + 3
        cv2.line(template, (cx, vertical_start_y), (cx, vertical_end_y), 255, self.template_thickness)

        # 水平段：从中心到右边
        horizontal_start_x = cx
        horizontal_end_x = w - 8
        cv2.line(template, (horizontal_start_x, vertical_end_y), (horizontal_end_x, vertical_end_y), 255,
                 self.template_thickness)

        # 箭头头部：指向右边
        head_tip_x = min(w - 2, horizontal_end_x + head_size // 2)
        head_top_y = vertical_end_y - head_size // 2
        head_bottom_y = vertical_end_y + head_size // 2

        # 绘制三角形箭头头部
        points = np.array([[head_tip_x, vertical_end_y],
                           [horizontal_end_x - head_size // 2, head_top_y],
                           [horizontal_end_x - head_size // 2, head_bottom_y]], np.int32)
        cv2.fillPoly(template, [points], 255)

        return template

    def _extract_template_features(self):
        """
        提取模板特征
        """
        if self.debug:
            print("开始提取模板特征...")

        for template_name, template in self.templates.items():
            keypoints, descriptors = self.orb.detectAndCompute(template, None)

            if descriptors is not None:
                self.template_features[template_name] = {
                    'keypoints': keypoints,
                    'descriptors': descriptors,
                    'template': template
                }

                if self.debug:
                    print(f"模板 {template_name} 提取到 {len(keypoints)} 个特征点")
            else:
                if self.debug:
                    print(f"模板 {template_name} 未检测到特征点")

    def _template_name_to_direction(self, template_name: str) -> ArrowDirection:
        """将模板名称转换为方向"""
        if 'straight' in template_name.lower():
            return ArrowDirection.STRAIGHT
        elif 'left' in template_name.lower():
            return ArrowDirection.LEFT
        elif 'right' in template_name.lower():
            return ArrowDirection.RIGHT
        else:
            return ArrowDirection.UNKNOWN

    def detect_arrows(self, image: np.ndarray) -> List[Tuple[ArrowDirection, np.ndarray, Tuple[int, int, int, int]]]:
        """
        检测图像中的箭头
        :param image: 输入图像
        :return: 检测结果列表，每个元素包含 (方向, 轮廓, 边界框)
        """
        # 预处理图像
        binary_image, roi_coords = self.preprocess_image(image)

        # 查找轮廓
        contours = self.find_arrow_contours(binary_image)

        results = []
        for i, contour in enumerate(contours):
            if self.debug:
                print(f"\n=== 分析轮廓 {i + 1}/{len(contours)} ===")

            # 使用新的模板匹配和特征匹配方法分类箭头方向
            direction = self.classify_arrow_direction(contour)

            if direction != ArrowDirection.UNKNOWN:
                # 获取边界框（需要转换回原图坐标）
                x, y, w, h = cv2.boundingRect(contour)
                # 转换为原图坐标
                x += roi_coords[0]
                y += roi_coords[1]

                # 同样转换轮廓坐标
                contour_original = contour.copy()
                contour_original[:, :, 0] += roi_coords[0]
                contour_original[:, :, 1] += roi_coords[1]

                results.append((direction, contour_original, (x, y, w, h)))

                if self.debug:
                    print(f"检测到箭头: {direction.value}")
            else:
                if self.debug:
                    print(f"轮廓未被识别为箭头")

        return results
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

        cv2.putText(image,
                    f"Left: {info['left_count']} | Right: {info['right_count']} | 直行: {info['straight_count']}",
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
    print(f"\n{'=' * 50}")
    print("FINAL STATISTICS RESULTS")
    print(f"{'=' * 50}")

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

    print(f"{'=' * 50}")





def detect_most_frequent_arrow(duration: float = 10.0, show_gui: bool = False, camera_id: int = 0) -> str:
    """
    在指定时间内持续检测箭头，并返回出现次数最多的方向。

    :param duration: 检测持续时间（秒）。
    :param show_gui: 是否显示GUI界面。
    :param camera_id: 要使用的摄像头ID。
    :return: 出现最多的箭头方向字符串 ('LEFT', 'RIGHT', 'STRAIGHT') 或 'UNKNOWN'。
    """
    print(f"--- 开始箭头检测，持续时间: {duration}秒 ---")

    # 1. 初始化组件
    camera = TransbotCamera(debug=True)
    detector = ArrowDetector(debug=False)
    total_counts = Counter()

    # 2. 初始化摄像头
    if not camera.initialize():
        print("错误: 摄像头初始化失败。")
        return "ERROR: Camera initialization failed."

    # 3. 开始检测循环
    start_time = time.time()
    try:
        while time.time() - start_time < duration:
            # 获取摄像头帧
            ret, frame = camera.get_camera_frame()
            if not ret or frame is None:
                print("警告: 无法读取摄像头帧。")
                time.sleep(0.1)
                continue

            # 检测箭头
            results = detector.detect_arrows(frame)

            # 更新统计
            for direction, _, _ in results:
                if direction != ArrowDirection.UNKNOWN:
                    total_counts[direction] += 1

            # 如果需要，显示GUI
            if show_gui:
                display_frame = frame.copy()
                # 绘制检测结果
                for direction, contour, bbox in results:
                    x, y, w, h = bbox
                    color = (255, 0, 0) if direction == ArrowDirection.LEFT else \
                        (0, 0, 255) if direction == ArrowDirection.RIGHT else \
                            (0, 255, 0) if direction == ArrowDirection.STRAIGHT else \
                                (128, 128, 128)
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(display_frame, direction.value, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # 添加状态信息
                _add_detection_overlay(display_frame, time.time() - start_time, duration, total_counts)
                cv2.imshow('Arrow Detection', display_frame)

                # 按 'q' 退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("用户提前退出。")
                    break

    except KeyboardInterrupt:
        print("用户中断了检测。")
    finally:
        # 4. 清理资源
        print("--- 检测结束，正在清理资源 ---")
        camera.close_camera()
        if show_gui:
            cv2.destroyAllWindows()

    # 5. 计算并返回结果
    print("\n--- 最终统计结果 ---")
    if not total_counts:
        print("未检测到任何有效箭头。")
        return ArrowDirection.UNKNOWN.name

    # 打印所有检测到的箭头及其计数
    for direction, count in total_counts.items():
        print(f"  {direction.value}: {count} 次")

    # 找到最常见的箭头
    most_common_direction, count = total_counts.most_common(1)[0]
    print(f"\n出现最多的箭头是: {most_common_direction.value} (共 {count} 次)")

    return most_common_direction.name


def _add_detection_overlay(image, elapsed_time, total_duration, counts):
    """在图像上添加一个简单的信息覆盖层"""
    h, w, _ = image.shape

    # 绘制半透明背景
    overlay = image.copy()
    cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    # 准备文本
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 时间和进度
    progress = (elapsed_time / total_duration) * 100
    time_text = f"Time: {elapsed_time:.1f}s / {total_duration:.0f}s ({progress:.1f}%)"
    cv2.putText(image, time_text, (10, h - 35), font, 0.6, (255, 255, 255), 2)

    # 统计计数
    left = counts.get(ArrowDirection.LEFT, 0)
    right = counts.get(ArrowDirection.RIGHT, 0)
    straight = counts.get(ArrowDirection.STRAIGHT, 0)
    count_text = f"Counts: L={left}, R={right}, S={straight}"
    cv2.putText(image, count_text, (10, h - 10), font, 0.6, (0, 255, 255), 2)


# 在主函数中添加一个测试入口
if __name__ == "__main__":
    import sys

    # 根据命令行参数选择运行模式
    if len(sys.argv) > 1 and sys.argv[1] == 'detect':
        # 运行新的检测函数
        # 示例: python ArrowMatching.py detect 15 nogui
        #       (检测15秒，不显示GUI)

        run_duration = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
        run_gui = False if len(sys.argv) > 3 and sys.argv[3] == 'nogui' else True

        final_result = detect_most_frequent_arrow(duration=run_duration, show_gui=run_gui)
        print(f"\n[函数返回值]: {final_result}")

    elif len(sys.argv) > 1 and sys.argv[1] == 'debug':
        # 运行原始的调试函数
        debug_arrow_statistics()
    else:
        # 默认运行新的检测函数
        print("默认模式: 运行10秒箭头检测 (带GUI)")
        print("使用 'python ArrowMatching.py debug' 运行完整调试版本")
        print("使用 'python ArrowMatching.py detect <秒数> [nogui]' 自定义运行")
        final_result = detect_most_frequent_arrow(duration=10.0, show_gui=True)
        print(f"\n[函数返回值]: {final_result}")