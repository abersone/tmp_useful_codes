import os
import cv2
import numpy as np
from pathlib import Path
import argparse

def parse_yolo11_segmentation_label(label_path, img_width, img_height):
    """
    解析YOLO11分割标签文件
    格式：N行M列，第一列为类别，其余列为归一化坐标点
    """
    contours = []
    classes = []
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            values = line.split()
            if len(values) < 3:  # 至少需要类别和两个坐标点
                continue
                
            try:
                class_id = int(values[0])
                classes.append(class_id)
                
                # 解析坐标点
                points = []
                for i in range(1, len(values), 2):
                    if i + 1 < len(values):
                        x_norm = float(values[i])
                        y_norm = float(values[i + 1])
                        
                        # 转换为像素坐标
                        x = int(x_norm * img_width)
                        y = int(y_norm * img_height)
                        points.append([x, y])
                
                if len(points) >= 3:  # 至少需要3个点形成轮廓
                    contours.append(np.array(points, dtype=np.int32))
                    
            except (ValueError, IndexError) as e:
                print(f"解析标签文件 {label_path} 时出错: {e}")
                continue
                
    except FileNotFoundError:
        print(f"标签文件不存在: {label_path}")
    except Exception as e:
        print(f"读取标签文件 {label_path} 时出错: {e}")
    
    return contours, classes

def draw_segmentation_contours(image, contours, classes, class_names=None):
    """
    在图像上绘制分割轮廓和类别标签
    """
    img_draw = image.copy()
    
    # 定义颜色映射（为不同类别分配不同颜色）
    colors = [
        (255, 0, 0),    # 蓝色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 红色
        (255, 255, 0),  # 青色
        (255, 0, 255),  # 洋红色
        (0, 255, 255),  # 黄色
        (128, 0, 0),    # 深蓝色
        (0, 128, 0),    # 深绿色
        (0, 0, 128),    # 深红色
        (128, 128, 0),  # 橄榄色
    ]
    
    for i, (contour, class_id) in enumerate(zip(contours, classes)):
        # 选择颜色
        color = colors[class_id % len(colors)]
        
        # 绘制轮廓
        cv2.drawContours(img_draw, [contour], -1, color, 2)
        
        # 计算轮廓中心点用于放置文本
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # 如果无法计算中心点，使用轮廓的第一个点
            cx, cy = contour[0][0], contour[0][1]
        
        # 准备类别标签文本
        if class_names and class_id < len(class_names):
            label = f"{class_names[class_id]}"
        else:
            label = f"{class_id}"
        
        # 绘制类别标签背景
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_draw, (cx - 5, cy - text_height - 10), (cx + text_width + 5, cy + 5), color, -1)
        
        # 绘制类别标签文本
        cv2.putText(img_draw, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img_draw

def process_dataset(input_folder, output_folder, class_names=None):
    """
    处理整个数据集，包括train和val文件夹
    文件夹结构: images/train, images/val, labels/train, labels/val
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # 创建输出文件夹结构
    for split in ['train', 'val']:
        (output_path / split).mkdir(parents=True, exist_ok=True)
    
    # 检查输入文件夹结构
    images_path = input_path / 'images'
    labels_path = input_path / 'labels'
    
    if not images_path.exists() or not labels_path.exists():
        print(f"错误: 输入文件夹缺少 images 或 labels 子文件夹")
        return
    
    # 全局统计变量
    total_class_count = {}  # 存储所有类别的总数量
    total_objects = 0  # 总对象数量
    total_images = 0  # 总图像数量
    
    # 处理train和val数据
    for split in ['train', 'val']:
        split_images_path = images_path / split
        split_labels_path = labels_path / split
        split_output_path = output_path / split
        
        if not split_images_path.exists():
            print(f"警告: {split_images_path} 不存在，跳过")
            continue
        
        if not split_labels_path.exists():
            print(f"警告: {split_labels_path} 不存在，跳过")
            continue
        
        print(f"处理 {split} 数据集...")
        
        # 获取所有图像文件
        image_files = list(split_images_path.glob('*.jpg')) + list(split_images_path.glob('*.png')) + list(split_images_path.glob('*.jpeg'))
        
        processed_count = 0
        split_class_count = {}  # 当前split的类别统计
        split_objects = 0  # 当前split的对象数量
        for img_file in image_files:
            # 构造对应的标签文件路径
            label_file = split_labels_path / f"{img_file.stem}.txt"
            
            if not label_file.exists():
                print(f"警告: 找不到对应的标签文件 {label_file}")
                continue
            
            try:
                # 读取图像
                image = cv2.imread(str(img_file))
                if image is None:
                    print(f"警告: 无法读取图像 {img_file}")
                    continue
                
                img_height, img_width = image.shape[:2]
                
                # 解析标签文件
                contours, classes = parse_yolo11_segmentation_label(label_file, img_width, img_height)
                
                # 统计类别数量
                for class_id in classes:
                    # 统计当前split的类别
                    if class_id in split_class_count:
                        split_class_count[class_id] += 1
                    else:
                        split_class_count[class_id] = 1
                    split_objects += 1
                    
                    # 统计全局类别
                    if class_id in total_class_count:
                        total_class_count[class_id] += 1
                    else:
                        total_class_count[class_id] = 1
                    total_objects += 1
                
                if contours:
                    # 绘制轮廓和标签
                    result_image = draw_segmentation_contours(image, contours, classes, class_names)
                else:
                    # 如果没有轮廓，直接使用原图
                    result_image = image
                
                # 保存结果图像
                output_img_path = split_output_path / img_file.name
                cv2.imwrite(str(output_img_path), result_image)
                
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"已处理 {split} 数据集中的 {processed_count} 张图像")
                    
            except Exception as e:
                print(f"处理图像 {img_file} 时出错: {e}")
                continue
        
        print(f"{split} 数据集处理完成，共处理 {processed_count} 张图像")
        
        # 打印当前split的统计信息
        if split_class_count:
            print(f"  {split} 数据集统计:")
            print(f"    总对象数量: {split_objects}")
            print(f"    发现类别数量: {len(split_class_count)}")
            sorted_classes = sorted(split_class_count.items())
            for class_id, count in sorted_classes:
                class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"类别{class_id}"
                percentage = (count / split_objects) * 100 if split_objects > 0 else 0
                print(f"      {class_name} (ID: {class_id}): {count} 个对象 ({percentage:.1f}%)")
        
        total_images += processed_count
    
    # 打印全局统计信息
    print("\n=== 全局统计信息 ===")
    print(f"总图像数量: {total_images}")
    if total_class_count:
        print(f"总对象数量: {total_objects}")
        print(f"发现类别数量: {len(total_class_count)}")
        print("\n各类别详细统计:")
        
        # 按类别ID排序
        sorted_classes = sorted(total_class_count.items())
        for class_id, count in sorted_classes:
            class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"类别{class_id}"
            percentage = (count / total_objects) * 100 if total_objects > 0 else 0
            print(f"  {class_name} (ID: {class_id}): {count} 个对象 ({percentage:.1f}%)")
    else:
        print("未发现任何标注对象")

def main():
    # 使用新的文件夹结构: images/train, images/val, labels/train, labels/val
    input_folder = r"C:\Users\Eugene\Desktop\vehicle_dataset\seg_ball_v1"
    output_folder = input_folder + "/vis"
    process_dataset(input_folder, output_folder)
    
    print("数据集可视化处理完成！")

if __name__ == "__main__":
    main()
