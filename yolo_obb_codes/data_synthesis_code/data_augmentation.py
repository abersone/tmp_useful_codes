import cv2
import numpy as np
import os
from pathlib import Path
import random
from tqdm import tqdm
import math
from copy import deepcopy
from get_mask import generate_mask
from copy_paste import blend_template

def flip_image_and_label(image, label_points, mode):
    """
    对图像和标签进行翻转和旋转
    Args:
        image: 输入图像
        label_points: 标签点坐标 [N, 8] 归一化坐标
        mode: 0-原图, 1-水平翻转, 2-上下翻转, 
             3-旋转90度, 4-旋转180度, 5-旋转270度
    Returns:
        transformed_image, transformed_points
    """
    h, w = image.shape[:2]
    
    if mode == 0:  # 原图
        return image.copy(), label_points.copy()
    
    elif mode == 1:  # 水平翻转
        transformed_image = cv2.flip(image, 1)
        transformed_points = label_points.copy()
        for i in range(0, len(transformed_points), 2):
            transformed_points[i] = 1.0 - transformed_points[i]
            
    elif mode == 2:  # 上下翻转
        transformed_image = cv2.flip(image, 0)
        transformed_points = label_points.copy()
        for i in range(1, len(transformed_points), 2):
            transformed_points[i] = 1.0 - transformed_points[i]
            
    elif mode == 3:  # 旋转90度
        transformed_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        transformed_points = []
        for i in range(0, len(label_points), 2):
            x, y = label_points[i], label_points[i+1]
            # 旋转90度的坐标变换：(x,y) -> (y, 1-x)
            transformed_points.extend([y, 1.0 - x])
            
    elif mode == 4:  # 旋转180度
        transformed_image = cv2.rotate(image, cv2.ROTATE_180)
        transformed_points = []
        for i in range(0, len(label_points), 2):
            x, y = label_points[i], label_points[i+1]
            # 旋转180度的坐标变换：(x,y) -> (1-x, 1-y)
            transformed_points.extend([1.0 - x, 1.0 - y])
            
    elif mode == 5:  # 旋转270度
        transformed_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        transformed_points = []
        for i in range(0, len(label_points), 2):
            x, y = label_points[i], label_points[i+1]
            # 旋转270度的坐标变换：(x,y) -> (1-y, x)
            transformed_points.extend([1.0 - y, x])
            
    else:
        raise ValueError(f"不支持的转换模式: {mode}")
    
    return transformed_image, np.array(transformed_points)

def visualize_rotated_box(image, points, color=(0, 255, 0), thickness=2):
    """
    可视化旋转框
    Args:
        image: 输入图像
        points: 归一化坐标 [8,]
    """
    h, w = image.shape[:2]
    img_points = []
    for i in range(0, len(points), 2):
        x = int(points[i] * w)
        y = int(points[i + 1] * h)
        img_points.append([x, y])
    
    img_points = np.array(img_points, dtype=np.int32)
    cv2.polylines(image, [img_points], True, color, thickness)
    return image

def check_box_overlap(box1, box2, image_size, iou_threshold=0.0):
    """
    检查两个旋转框是否重叠
    Args:
        box1, box2: 归一化坐标 [8,]
        image_size: (h, w)
        iou_threshold: IoU阈值
    """
    h, w = image_size
    
    def to_pixel_coords(points):
        pixel_points = []
        for i in range(0, len(points), 2):
            x = int(points[i] * w)
            y = int(points[i + 1] * h)
            pixel_points.append([x, y])
        return np.array(pixel_points, dtype=np.int32)
    
    pts1 = to_pixel_coords(box1)
    pts2 = to_pixel_coords(box2)
    
    # 创建mask
    mask1 = np.zeros((h, w), dtype=np.uint8)
    mask2 = np.zeros((h, w), dtype=np.uint8)
    
    cv2.fillPoly(mask1, [pts1], 1)
    cv2.fillPoly(mask2, [pts2], 1)
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return False
    
    iou = intersection / union
    return iou > iou_threshold

def check_box_distance(box1, box2, image_size, min_center_dist=100):
    """
    检查两个旋转框的中心点距离是否小于阈值
    Args:
        box1, box2: 归一化坐标 [8,]
        image_size: (h, w)
        min_center_dist: 两个框中心点的最小距离（像素），默认100
    Returns:
        bool: True表示中心点距离小于阈值
    """
    h, w = image_size    
    # 计算框的中心点
    def get_box_center(box):
        # 先转换为numpy数组确保一致的数据类型
        box = np.array(box, dtype=np.float64)
        # 分别获取x和y坐标
        x_coords = box[0::2]  # 所有x坐标（归一化）
        y_coords = box[1::2]  # 所有y坐标（归一化）
        
        # 计算归一化坐标的平均值
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        
        # 转换为像素坐标
        center_x = center_x * w
        center_y = center_y * h
        
        return center_x, center_y
    
    # 计算两个中心点的距离
    center1_x, center1_y = get_box_center(box1)
    center2_x, center2_y = get_box_center(box2)
    
    # 计算欧氏距离
    center_dist = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    
    # 如果中心点距离小于阈值，返回True
    return center_dist < min_center_dist

def detect_circle_center(img, draw=False):
    """
    检测图像中的大圆中心点
    Args:
        img: 输入图像
        draw: 是否在图像上绘制检测结果
    Returns:
        tuple: (center_x, center_y, radius) 如果检测成功，否则返回None
    """
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blurred = gray
    # 修改最小半径为图像宽度的1/4
    min_radius = int(img.shape[1] * 0.25)
    # 修改最大半径为图像宽度的0.4
    max_radius = int(img.shape[1] * 0.4)
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_radius*2,
        param1=50,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        largest_circle = circles[np.argmax(circles[:, 2])]
        center_x, center_y, radius = largest_circle
        
        if draw:
            cv2.circle(img, (center_x, center_y), 3, (0, 0, 255), -1)
            cv2.circle(img, (center_x, center_y), radius, (0, 255, 0), 2)
        
        return (center_x, center_y, radius)
    return (None, None, None)

def get_random_position(center, radius, radius_ranges, template_size, image_size):
    """
    在圆环区域内随机选择一个位置，确保返回有效位置
    Args:
        center: 圆心坐标 (x, y)
        radius: 圆半径
        radius_ranges: 半径范围的比例列表 [(s1, s2), (s3, s4), ...]
        template_size: 模板大小 (h, w)
        image_size: 目标图像大小 (h, w)
    Returns:
        tuple: ((x, y), selected_range_index) 返回位置坐标和所选范围的索引
    """
    th, tw = template_size
    image_h, image_w = image_size
    cx, cy = center
    
    # 参数合法性检查
    if center is None or radius is None:
        raise ValueError("圆心坐标或半径不能为None")
    if not (0 <= cx < image_w and 0 <= cy < image_h):
        raise ValueError(f"圆心坐标({cx}, {cy})超出图像范围")
    if radius <= 0:
        raise ValueError("半径必须为正数")
    
    # 验证所有范围的有效性
    for i, (s1, s2) in enumerate(radius_ranges):
        if not (0 <= s1 < s2):
            raise ValueError(f"radius_range {i} 必须满足: 0 <= s1 < s2，当前值: ({s1}, {s2})")
    
    min_required_space = max(th//2, tw//2)
    
    # 随机打乱范围的顺序，以随机选择一个可用范围
    range_indices = list(range(len(radius_ranges)))
    random.shuffle(range_indices)
    
    # 对每个范围尝试获取有效位置
    for range_idx in range_indices:
        s1, s2 = radius_ranges[range_idx]
        min_distance = radius * s1
        max_distance = radius * s2
        
        # 检查当前范围是否可能存在有效区域
        if max_distance + min_required_space > min(image_h//2, image_w//2):
            continue
        
        max_attempts = 1000  # 每个范围的最大尝试次数
        attempts = 0
        
        while attempts < max_attempts:
            # 随机选择角度和距离
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(min_distance, max_distance)
            
            # 计算位置（模板中心点）
            x = int(cx + distance * math.cos(angle))
            y = int(cy + distance * math.sin(angle))
            
            # 检查是否在有效范围内
            if (x - tw//2 >= 0 and x + tw//2 < image_w and 
                y - th//2 >= 0 and y + th//2 < image_h):
                return (x, y), range_idx
                
            attempts += 1
    
    return (-1, -1), -1  # 返回无效坐标和无效范围索引

def convert_template_box_to_target(template_box, template_size, pos, target_size):
    """
    将模板上的标注框转换到目标图像坐标系
    Args:
        template_box: 模板上的标注框坐标（归一化）
        template_size: 模板尺寸 (h, w)
        pos: 模板在目标图像上的位置 (x, y)（中心点）
        target_size: 目标图像尺寸 (h, w)
    Returns:
        target_box: 目标图像上的标注框坐标（归一化）
    """
    th, tw = template_size
    target_h, target_w = target_size
    px, py = pos
    
    # 转换到目标图像坐标系
    target_box = []
    for i in range(0, len(template_box), 2):
        # 从归一化坐标转换到像素坐标
        x = template_box[i] * tw
        y = template_box[i + 1] * th
        
        # 平移到目标图像上的位置
        x = x + px
        y = y + py
        
        # 转换回归一化坐标
        x = x / target_w
        y = y / target_h
        
        target_box.extend([x, y])
    
    return target_box

def check_mask_bbox_in_range(mask, pos, center, radius, radius_range):
    """
    检查mask的最小外接矩形在pos位置时是否完全在指定的圆环区域内
    
    Args:
        mask: 模板的mask
        pos: 模板在目标图像上的位置 (x, y)（中心点）
        center: 圆心坐标 (x, y)
        radius: 圆半径
        radius_range: 半径范围的比例 (s1, s2)
    
    Returns:
        bool: True表示矩形框完全在圆环区域内
    """
    # 找到mask中非零点的坐标
    non_zero = cv2.findNonZero(mask)
    if non_zero is None:
        return True
    
    # 计算最小外接矩形
    x, y, w, h = cv2.boundingRect(non_zero)
    
    # 计算矩形的四个角点（相对于模板坐标系）
    corners = np.array([
        [x, y],           # 左上
        [x + w, y],       # 右上
        [x + w, y + h],   # 右下
        [x, y + h]        # 左下
    ])
    
    # 将角点转换到目标图像坐标系
    corners[:, 0] += pos[0]# - mask.shape[1]//2
    corners[:, 1] += pos[1]# - mask.shape[0]//2
    
    # 计算每个角点到圆心的距离
    cx, cy = center
    distances = np.sqrt(np.sum((corners - np.array([cx, cy]))**2, axis=1))
    
    # 检查所有角点是否都在圆环区域内
    min_radius = radius * radius_range[0]
    max_radius = radius * radius_range[1]
    
    return np.all((distances >= min_radius) & (distances <= max_radius))

# main function
def process_templates(template_dir, template_label_dir, target_image_dir, target_label_dir,
                     mask_dir,
                     sample_ratio=0.1, radius_ranges=[(0.5, 0.8), (1.05, 1.3)], max_attempts=100,
                     mask_cut_lines=0, mask_std_ratio=0.5,
                     blend_dilate_size=5,
                     debug=False, debug_dir=None, debug_template_index=0):
    """
    处理模板并直接进行粘贴操作
    Args:
        template_dir: 模板图像目录
        template_label_dir: 模板标签目录
        target_image_dir: 目标图像目录
        target_label_dir: 目标标签目录
        mask_dir: mask图像目录
        sample_ratio: 目标图像采样比例
        radius_ranges: 圆环区域的半径范围比例列表，每个元素为(内圈比例, 外圈比例)
        max_attempts: 最大尝试次数
        mask_cut_lines: 生成mask时的线条数量
        mask_std_ratio: 生成mask时的标准差比例
        blend_dilate_size: mask膨胀的核大小
        debug: 是否开启调试模式
        debug_dir: 调试文件保存目录，如果为None则默认为'debug'
        debug_template_index: 调试模板索引
    """
    # 创建debug目录
    if debug:
        debug_dir = debug_dir or 'debug'
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(exist_ok=True)
        template_debug_dir = debug_dir / 'templates'
        template_debug_dir.mkdir(exist_ok=True)
    
    # 1. 获取所有模板和目标文件
    template_files = list(Path(template_dir).glob('*.jpg'))
    target_files = list(Path(target_image_dir).glob('*.jpg'))
    
    print(f"模板文件数量: {len(template_files)}")
    print(f"目标文件总数: {len(target_files)}")
    print(f"每个模板选择比例: {sample_ratio:.2%}")
    if debug:
        # 使用传入的debug_template_index
        if debug_template_index < len(template_files):
            template_files = [template_files[debug_template_index]]
        else:
            raise ValueError(f"指定的模板序号 {debug_template_index} 超出范围，总模板数量为 {len(template_files)}")
    
    # 取前30个模板文件
    # template_files = template_files[:30] # 调试时使用
    print(f"实际使用的模板文件数量: {len(template_files)}")
    
    # 2. 处理每个模板
    for template_file in tqdm(template_files, desc="处理模板"):
        # 2.1 读取原始模板图像
        orig_template = cv2.imread(str(template_file))
        if orig_template is None:
            continue
            
        # 2.2 读取对应的mask文件
        mask_file = Path(mask_dir) / f"{template_file.stem}_mask.png"
        if not mask_file.exists():
            print(f"警告: 未找到对应的mask文件: {mask_file}")
            continue
        
        orig_mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if orig_mask is None:
            print(f"警告: 无法读取mask文件: {mask_file}")
            continue
        # 对原始mask进行5x5膨胀和腐蚀, 抹平mask边缘毛刺
        kernel = np.ones((5,5), np.uint8)
        orig_mask = cv2.dilate(orig_mask, kernel, iterations=1)
        orig_mask = cv2.erode(orig_mask, kernel, iterations=1)
        kernel = np.ones((3,3), np.uint8)
        orig_mask = cv2.erode(orig_mask, kernel, iterations=1)
        # 2.3 读取原始模板标签
        label_file = Path(template_label_dir) / f"{template_file.stem}.txt"
        if not label_file.exists():
            continue
            
        with open(label_file, 'r') as f:
            template_label = f.readline().strip()
        template_cls = int(template_label.split()[0])
        orig_points = np.array([float(x) for x in template_label.split()[1:]])
        
        # 2.4 可视化原始模板的标签
        if debug:
            debug_template = orig_template.copy()
            visualize_rotated_box(debug_template, orig_points)
            debug_save_path = template_debug_dir / f"{template_file.stem}_original_tmpl.jpg"
            cv2.imwrite(str(debug_save_path), debug_template)
        
        # 2.5 对每个模板进行旋转和翻转预处理
        for transform_mode in range(5):  # 0-原图, 1-水平翻转, 2-上下翻转, 3-90度, 4-180度, 5-270度
            # 2.5.1 获取变换后的模板和标签
            template, points = flip_image_and_label(orig_template, orig_points, transform_mode)
            
            # 可视化变换后的模板和标签
            if debug:
                debug_template = template.copy()
                visualize_rotated_box(debug_template, points)
                transform_names = ['orig', 'flip_h', 'flip_v', '90', '180', '270']
                debug_save_path = template_debug_dir / f"{template_file.stem}_{transform_names[transform_mode]}_tmpl.jpg"
                cv2.imwrite(str(debug_save_path), debug_template)
                
            # # 生成mask
            # mask = generate_mask(
            #     template=template, 
            #     cut_lines=mask_cut_lines, 
            #     std_ratio=mask_std_ratio, 
            #     debug=False, 
            #     debug_dir=debug_dir
            # )
            # 对mask进行相同的变换
            mask, _ = flip_image_and_label(orig_mask, orig_points, transform_mode)
            
            # 可视化变换后的mask
            if debug:
                debug_mask = mask.copy()
                transform_names = ['orig', 'flip_h', 'flip_v', '90', '180', '270']
                debug_save_path = template_debug_dir / f"{template_file.stem}_{transform_names[transform_mode]}_mask.jpg"
                cv2.imwrite(str(debug_save_path), debug_mask)
                
            # 2.5.2 随机选择目标图像
            # 在程序开始时设置随机种子
            random.seed(None)  # 使用系统时间作为种子
            # 或者使用特定的种子值
            # random.seed(42)  # 使用固定的种子值，用于复现结果
            # debug模式下的特殊处理
            if debug:
                # 限制目标图像数量为2
                max_debug_targets = 2
                if len(target_files) > max_debug_targets:
                    # 可以选择随机采样或者取前10张
                    selected_targets = random.sample(target_files, max_debug_targets)
                    # 或者使用前2张
                    # selected_targets = target_files[:max_debug_targets]
                else:
                    selected_targets = target_files
                    
                print(f"Debug模式: 使用模板 {template_files[0].name}")
                print(f"Debug模式: 处理 {len(selected_targets)} 张目标图像")
            else:
                # 原有的目标图像选择逻辑
                num_samples = int(len(target_files) * sample_ratio)
                if num_samples == len(target_files):
                    selected_targets = target_files
                else:
                    selected_targets = random.sample(target_files, num_samples)
            
            print(f"模板: {template_file.name}, 模板变换类型: {transform_mode}")
            print(f"选择的目标文件数量: {len(selected_targets)}")
            
            
            # 2.5.3 处理每个选定的目标图像
            for target_file in tqdm(selected_targets, desc="处理目标图像", leave=False):
                target_img = cv2.imread(str(target_file))
                if target_img is None:
                    continue
                    
                # 检测圆心和半径
                center_x, center_y, radius = detect_circle_center(target_img.copy())
                if None in (center_x, center_y, radius):
                    continue
                    
                # 读取目标图像的现有标签
                target_label_file = Path(target_label_dir) / f"{target_file.stem}.txt"
                existing_boxes = []
                if target_label_file.exists():
                    with open(target_label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            cls_id = int(float(parts[0]))
                            box_points = [float(x) for x in parts[1:]]
                            existing_boxes.append((cls_id, box_points))
                
                # 尝试不同位置直到找到无重叠的位置或达到最大尝试次数
                for _ in range(max_attempts):
                    # 获取一个随机位置和选中的范围索引
                    pos, range_idx = get_random_position(
                        (center_x, center_y), 
                        radius, 
                        radius_ranges,  # 传入范围列表
                        template.shape[:2],
                        target_img.shape[:2]
                    )

                    # 检查pos是否有效
                    if pos[0] < 0 or pos[1] < 0:
                        continue
                    if pos[0] + template.shape[1] > target_img.shape[1] or pos[1] + template.shape[0] > target_img.shape[0]:
                        continue
                    
                    # 转换模板框到目标图像坐标系
                    target_box = convert_template_box_to_target(
                        points,
                        template.shape[:2],
                        pos,
                        target_img.shape[:2]
                    )
                    
                    # 检查mask的最小外接矩形是否在有效区域内
                    if range_idx >= 0:  # 确保找到了有效的范围
                        if not check_mask_bbox_in_range(
                            mask, 
                            pos, 
                            (center_x, center_y), 
                            radius, 
                            radius_ranges[range_idx]  # 使用选中的范围
                        ):
                            continue
                    else:
                        continue
                    
                    # 检查是否与现有框重叠
                    has_overlap = False
                    for _, existing_box in existing_boxes:
                        if check_box_overlap(target_box, existing_box, target_img.shape[:2]):
                            if 0: #debug:
                                # 创建一个副本用于可视化
                                debug_overlap = target_img.copy()
                                # 绘制target_box(红色)
                                visualize_rotated_box(debug_overlap, target_box, color=(0,0,255), thickness=2)
                                # 绘制existing_box(绿色)
                                visualize_rotated_box(debug_overlap, existing_box, color=(0,255,0), thickness=2)
                                # 保存重叠框的可视化结果
                                overlap_save_path = debug_dir / f"{target_file.stem}_overlap_{_}.jpg"
                                cv2.imwrite(str(overlap_save_path), debug_overlap)
                            has_overlap = True
                            break
                        if check_box_distance(target_box, existing_box, target_img.shape[:2], min_center_dist=150):
                            has_overlap = True
                            break
                    
                    if has_overlap:  # 如果有重叠，继续下一次循环
                        continue

                    # 找到了无重叠的位置，进行图像融合和保存
                    result = blend_template(
                        template=template,
                        mask=mask,
                        target_img=target_img,
                        pos=pos,
                        blur_size=np.random.randint(25, 45),
                        opacity=np.random.uniform(0.86, 1.0),
                        dilate_size= 2 * np.random.randint(2, blend_dilate_size // 2) + 1, # -1,
                        save_process=False,
                        debug_dir=debug_dir
                    )
                    
                    # 保存结果图像
                    cv2.imwrite(str(target_file), result)
                    
                    # 读取现有标签
                    existing_content = ""
                    if os.path.exists(target_label_file):
                        with open(target_label_file, 'r') as f:
                            existing_content = f.read().rstrip()  # 移除末尾的空行
                    
                    # 写入更新后的标签
                    with open(target_label_file, 'w') as f:
                        label_line = f"{template_cls} " + " ".join([f"{x:.6f}" for x in target_box])
                        if existing_content:
                            f.write(f"{existing_content}\n{label_line}")
                        else:
                            f.write(label_line)
                    
                    if debug:
                        # 重新读取图像和标签进行验证
                        debug_img = cv2.imread(str(target_file))
                        # 在debug图像上绘制检测到的圆
                        cv2.circle(debug_img, (center_x, center_y), 3, (0, 0, 255), -1)  # 圆心
                        cv2.circle(debug_img, (center_x, center_y), radius, (0, 255, 0), 2)  # 圆周
                        
                        # 读取标签文件的所有行
                        with open(target_label_file, 'r') as f:
                            lines = f.readlines()
                            
                        # 遍历所有行，除了最后一行使用绿色，最后一行（新合成的）使用红色
                        for i, line in enumerate(lines):
                            parts = line.strip().split()
                            cls_id = int(float(parts[0]))
                            draw_points = [float(x) for x in parts[1:]]
                            
                            # 最后一行（新合成的标签）使用黄色，其他使用绿色
                            color = (0, 255, 255) if i == len(lines)-1 else (0, 255, 0)
                            visualize_rotated_box(debug_img, draw_points, color=color)
                            
                        cv2.imwrite(str(debug_dir / f"{target_file.stem}_debug.jpg"), debug_img)
                    
                    break  # 完成所有操作后退出循环
                
    # 在非debug模式下,完成所有循环后进行可视化
    if not debug:
        # 创建visualization目录
        vis_dir = Path(debug_dir) / 'visualization'
        vis_dir.mkdir(exist_ok=True)
        
        print("正在对所有目标图像进行可视化...")
        # 遍历所有目标图像
        for target_file in tqdm(target_files, desc="可视化处理"):
            # 读取图像
            vis_img = cv2.imread(str(target_file))
            if vis_img is None:
                continue
                
            # 读取对应的标签文件
            target_label_file = Path(target_label_dir) / f"{target_file.stem}.txt"
            if not target_label_file.exists():
                continue
                
            # 检测圆心和半径用于可视化
            center_x, center_y, radius = detect_circle_center(vis_img.copy())
            if None not in (center_x, center_y, radius):
                # 绘制检测到的圆
                cv2.circle(vis_img, (center_x, center_y), 3, (0, 0, 255), -1)  # 圆心
                cv2.circle(vis_img, (center_x, center_y), radius, (0, 255, 0), 2)  # 圆周
            
            # 读取并绘制所有标签框
            with open(target_label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    cls_id = int(float(parts[0]))
                    box_points = [float(x) for x in parts[1:]]
                    # 使用绿色绘制所有标签框
                    visualize_rotated_box(vis_img, box_points, color=(0, 255, 0))
            
            # 保存可视化结果
            vis_save_path = vis_dir / f"{target_file.stem}_vis.jpg"
            cv2.imwrite(str(vis_save_path), vis_img)
            
        print(f"可视化完成，结果保存在: {vis_dir}")
    return

def main():
    # 设置路径
    # dark dirt
    template_dir = "/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/liuqin_data/dirt_tmpls/bg_dark"
    mask_dir = "/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/liuqin_data/dirt_tmpls/bg_dark_with_mask"
    radius_ranges = [
        (0.01, 0.4),   # 第一个范围
        (1.05, 1.3)   # 第二个范围
    ]
    
    # light dirt
    # template_dir = "/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/liuqin_data/dirt_tmpls/bg_light"
    # mask_dir = "/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/liuqin_data/dirt_tmpls/bg_light_with_mask"
    # radius_ranges = [
    # (0.5, 0.95)
    # ]
    
    # labels
    template_label_dir = "/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/liuqin_data/dirt_tmpls/labels"
    
    # target images
    target_image_dir = "/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/liuqin_data/train_data_sampling_yolo/images/val"
    target_label_dir = "/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/liuqin_data/train_data_sampling_yolo/labels/val"
    
    # debug output dir
    debug_dir = "/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/liuqin_data/debug_output"
    
    # 参数设置
    debug = False
    debug_template_index = 0
        
    sample_ratio = 0.006 # 0.02

    max_attempts = 100
    
    # mask生成参数
    mask_cut_lines = 0
    mask_std_ratio = 0.5
    
    # 图像融合膨胀最大值
    blend_dilate_size = 9
    
    # 合成数据
    process_templates(
        template_dir,
        template_label_dir,
        target_image_dir,
        target_label_dir,
        mask_dir,
        sample_ratio=sample_ratio,
        radius_ranges=radius_ranges,
        max_attempts=max_attempts,
        mask_cut_lines=mask_cut_lines,
        mask_std_ratio=mask_std_ratio,
        blend_dilate_size=blend_dilate_size,
        debug=debug,
        debug_dir=debug_dir,
        debug_template_index=debug_template_index
    )

if __name__ == "__main__":
    main()