import os
import cv2
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon
import shutil
from tqdm import tqdm

def calculate_area_and_contrast(img, coords):
    """计算给定旋转框的面积和对比度"""
    # 将一维坐标数组重塑为N×2的点坐标数组
    coords = np.array(coords).reshape(-1, 2)
    
    # 计算面积
    area = cv2.contourArea(coords.astype(np.float32))
    
    # 计算对比度
    # 创建mask
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [coords.astype(np.int32)], 1)
    
    # 计算目标区域的最大亮度
    target_pixels = img[mask == 1]
    if len(target_pixels) > 0:
        target_brightness_max = np.max(target_pixels)
    else:
        return area, 0
    
    # 计算周边区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilated = cv2.dilate(mask, kernel, iterations=2)
    border = cv2.subtract(dilated, mask)
    
    # 计算周边区域的亮度中值
    border_pixels = img[border == 1]
    if len(border_pixels) > 0:
        border_brightness_median = np.median(border_pixels)
        # 计算对比度
        contrast = abs(float(target_brightness_max) - float(border_brightness_median))
    else:
        contrast = 0
        
    return area, contrast

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
    
    # 使用高斯模糊减少噪声
    #blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    blurred = gray
    # 修改最小半径为图像宽度的1/4
    min_radius = int(img.shape[1] * 0.25)
    # 修改最大半径为图像宽度的0.4（确保直径小于图像宽度的0.8）
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
        # 获取最大的圆
        largest_circle = circles[np.argmax(circles[:, 2])]
        center_x, center_y, radius = largest_circle
        
        if draw:
            # 绘制圆心（红色）
            cv2.circle(img, (center_x, center_y), 3, (0, 0, 255), -1)
            # 绘制圆轮廓（绿色）
            cv2.circle(img, (center_x, center_y), radius, (0, 255, 0), 2)
        
        return (center_x, center_y, radius)
    return (None, None, None)

def draw_box_with_metrics(img, coords, color, area, contrast=None):
    """在图像上绘制框和指标"""
    coords = coords.astype(np.int32)
    cv2.polylines(img, [coords], True, color, 2)
    
    # 计算文本位置（使用框的左上角）
    text_x = np.min(coords[:, 0])
    text_y = np.min(coords[:, 1])
    
    # 绘制指标文本
    if contrast is not None:
        text = f"A:{area:.1f}\nC:{contrast:.1f}"
    else:
        text = f"A:{area:.1f}"
    
    # 分行绘制文本
    y_offset = 0
    text_x = text_x + 15
    for line in text.split('\n'):
        y = text_y + y_offset
        cv2.putText(img, line, (text_x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y_offset += 25

def process_single_folder(folder_path):
    """
    处理单个文件夹中的所有jpg图片，检查圆心提取情况
    
    Args:
        folder_path (str): 图片文件夹路径
    """
    folder_path = Path(folder_path)
    failed_images = []
    
    # 获取所有jpg图片
    image_files = list(folder_path.glob('*.jpg'))
    
    print(f"开始处理文件夹: {folder_path}")
    print(f"共发现 {len(image_files)} 张图片")
    
    for img_path in tqdm(image_files, desc='处理图片'):
        # 读取图像
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"无法读取图片: {img_path}")
            failed_images.append(str(img_path))
            continue
            
        # 进行中值滤波
        img = cv2.medianBlur(img, 3)
        
        # 尝试检测圆心
        center_x, center_y, radius = detect_circle_center(img)
        
        if center_x is None:
            print(f"未能检测到圆心: {img_path}")
            failed_images.append(str(img_path))
    
    # 输出统计信息
    print("\n处理完成！")
    print(f"总图片数: {len(image_files)}")
    print(f"检测失败数: {len(failed_images)}")
    print(f"失败率: {len(failed_images)/len(image_files)*100:.2f}%")
    
    if failed_images:
        print("\n以下图片检测失败:")
        for img in failed_images:
            print(img)

def process_dataset(base_dir, inner_area_thres, inner_cons_thres, 
                   outer_area_thres, outer_cons_thres, test_mode=False, test_num=20):
    """处理数据集"""
    # 简化输出目录，直接在base_dir下创建
    base_path = Path(base_dir)
    output_labels = base_path / 'labels_modified'
    output_vis = base_path / 'visualizations_modified'
    
    # 创建输出目录
    for dir_path in [output_labels / 'train', output_labels / 'val',
                     output_vis / 'train', output_vis / 'val']:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 颜色定义 (BGR格式)
    COLORS = {
        'kept_0': (0, 255, 0),    # 蓝色
        'kept_1': (255, 0, 0),  # 紫色
        'kept_2': (0, 0, 255),    # 绿色
        'deleted': (0, 255, 255)   # 黄色
    }
    
    # 处理训练集和验证集
    for split in ['train', 'val']:
        images_dir = base_path / 'images' / split
        labels_dir = base_path / 'labels' / split
        current_output_labels = output_labels / split
        current_output_vis = output_vis / split
        
        # 获取所有图像文件
        image_files = list(images_dir.glob('*.jpg'))
        # 只取前10个图像文件
        if test_mode:
            image_files = image_files[:test_num]

        for img_path in tqdm(image_files, desc=f'Processing {split} set'):
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
                
            # 读取图像并进行中值滤波
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.medianBlur(img, 3)
            
            # 创建可视化图像
            vis_img = np.copy(img)
            # 创建拼接图像
            concat_img = np.zeros((img.shape[0], img.shape[1]*2, 3), dtype=np.uint8)
            concat_img[:, :img.shape[1]] = img  # 左侧原图
            concat_img[:, img.shape[1]:] = img   # 右侧处理后的图
            
            try:
                labels = np.loadtxt(str(label_path))
                if len(labels.shape) == 1:
                    labels = labels.reshape(1, -1)
            except:
                continue
            
            kept_labels = []
            # 获取圆心和半径信息
            center_x, center_y, radius = detect_circle_center(img)
            if center_x is None:
                print(f"No circle detected in {img_path}")
                continue
            #continue
            radius = radius + 250
            # 在拼接图像的两侧都绘制圆心和轮廓
            # 左侧图像
            cv2.circle(concat_img[:, :img.shape[1]], (center_x, center_y), 3, (0, 0, 255), -1)  # 红色圆心
            cv2.circle(concat_img[:, :img.shape[1]], (center_x, center_y), radius, (0, 255, 0), 2)  # 绿色轮廓
            
            # 处理每个标签
            for label in labels:
                class_id = int(label[0])
                coords = label[1:].reshape(-1, 2)
                
                # 转换为绝对坐标
                abs_coords = coords.copy()
                abs_coords[:, 0] *= img.shape[1]
                abs_coords[:, 1] *= img.shape[0]
                
                if class_id == 0:
                    # 计算标签中心点
                    label_center = np.mean(abs_coords, axis=0)
                    distance = np.sqrt(
                        (label_center[0] - center_x)**2 + 
                        (label_center[1] - center_y)**2
                    )
                    current_area_thres = inner_area_thres if distance < radius else outer_area_thres
                    current_cons_thres = inner_cons_thres if distance < radius else outer_cons_thres
                    
                    # 计算面积和对比度
                    area, contrast = calculate_area_and_contrast(img, abs_coords)
                    
                    # 在左侧图像上绘制
                    if area >= current_area_thres and contrast >= current_cons_thres:
                        kept_labels.append(label)
                        draw_box_with_metrics(concat_img, 
                                           abs_coords, COLORS['kept_0'], 
                                           area, contrast)
                    else:
                        draw_box_with_metrics(concat_img, 
                                           abs_coords, COLORS['deleted'], 
                                           area, contrast)
                else:
                    # 非类别0的标签
                    kept_labels.append(label)
                    color = COLORS[f'kept_{class_id}'] if class_id in [1, 2] else COLORS['kept_0']
                    cv2.polylines(concat_img, 
                                [abs_coords.astype(np.int32)], 
                                True, color, 2)
            
            # 绘制标签
            for label in kept_labels:
                class_id = int(label[0])
                coords = label[1:].reshape(-1, 2)
                abs_coords = coords.copy()
                abs_coords[:, 0] *= img.shape[1]
                abs_coords[:, 1] *= img.shape[0]
                
                color = COLORS[f'kept_{class_id}'] if class_id in [0, 1, 2] else COLORS['kept_0']
                if class_id == 0:
                    area, contrast = calculate_area_and_contrast(img, abs_coords)
                    draw_box_with_metrics(concat_img, 
                                       abs_coords, color, area, contrast)
                else:
                    cv2.polylines(concat_img, 
                                [abs_coords.astype(np.int32)], 
                                True, color, 2)
            
            # 保存结果
            if kept_labels:
                # 保存过滤后的标签
                output_label_path = current_output_labels / f"{img_path.stem}.txt"
                np.savetxt(str(output_label_path), kept_labels, fmt='%f')
                
                # 保存可视化结果
                output_vis_path = current_output_vis / f"{img_path.stem}.jpg"
                cv2.imwrite(str(output_vis_path), concat_img)

def main():
    # 检测图像提取Hough 的失败率
    if False:
        folder_path = "/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/corner_case"
        process_single_folder(folder_path)
        return
    
    # 设置路径和参数
    base_dir = "/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/train_data/after_check/train_data_v6.1_yolo"
    test_mode = False
    test_num = 20
    # 内部区域的阈值（靠近圆心）
    inner_area_thres = 25
    inner_cons_thres = 20
    
    # 外部区域的阈值
    outer_area_thres = 50
    outer_cons_thres = 100
    

    # 处理数据集（移除output_dir参数）
    process_dataset(base_dir, 
                   inner_area_thres, inner_cons_thres,
                   outer_area_thres, outer_cons_thres,
                   test_mode=test_mode, test_num=test_num)
    
    

if __name__ == "__main__":
    main()