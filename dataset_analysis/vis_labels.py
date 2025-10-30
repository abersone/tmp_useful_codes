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

def find_matching_label(image_file, label_folder):
    """
    在标签文件夹中查找与图像文件匹配的标签文件
    匹配规则：图像文件名前缀与标签文件名前缀相同
    """
    image_stem = image_file.stem
    label_folder_path = Path(label_folder)
    
    # 查找所有txt文件
    label_files = list(label_folder_path.glob('*.txt'))
    
    for label_file in label_files:
        if label_file.stem == image_stem:
            return label_file
    
    return None

def process_images_and_labels(image_folder, label_folder, output_folder,class_names=None):
    """
    处理图像和标签文件，进行可视化
    """
    image_path = Path(image_folder)
    label_path = Path(label_folder)
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 检查输入文件夹是否存在
    if not image_path.exists():
        print(f"错误: 图像文件夹不存在: {image_folder}")
        return
    
    if not label_path.exists():
        print(f"错误: 标签文件夹不存在: {label_folder}")
        return
    
    # 获取所有图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_path.glob(ext)))
    
    # 去重，避免同一文件被重复添加
    image_files = list(set(image_files))
    
    if not image_files:
        print(f"警告: 在 {image_folder} 中未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    processed_count = 0
    matched_count = 0
    
    # 统计类别数量
    class_count = {}  # 存储每个类别的数量
    total_objects = 0  # 总对象数量
    
    for img_file in image_files:
        # 查找匹配的标签文件
        label_file = find_matching_label(img_file, label_folder)
        
        if label_file is None:
            print(f"警告: 未找到与 {img_file.name} 匹配的标签文件")
            continue
        
        matched_count += 1
        
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
                if class_id in class_count:
                    class_count[class_id] += 1
                else:
                    class_count[class_id] = 1
                total_objects += 1
            
            if contours:
                # 绘制轮廓和标签
                result_image = draw_segmentation_contours(image, contours, classes, class_names)
            else:
                # 如果没有轮廓，直接使用原图
                result_image = image
            
            # 保存结果图像
            output_img_path = Path(output_folder) / img_file.name
            cv2.imwrite(str(output_img_path), result_image)
            
            processed_count += 1
            
            if processed_count % 50 == 0:
                print(f"已处理 {processed_count} 张图像")
                
        except Exception as e:
            print(f"处理图像 {img_file} 时出错: {e}")
            continue
    
    print(f"处理完成！")
    print(f"总共找到 {len(image_files)} 个图像文件")
    print(f"成功匹配 {matched_count} 个标签文件")
    print(f"成功处理 {processed_count} 张图像")
    print(f"可视化结果保存在: {output_folder}")
    
    # 打印类别统计信息
    print("\n=== 类别统计信息 ===")
    if class_count:
        print(f"总对象数量: {total_objects}")
        print(f"发现类别数量: {len(class_count)}")
        print("\n各类别详细统计:")
        
        # 按类别ID排序
        sorted_classes = sorted(class_count.items())
        for class_id, count in sorted_classes:
            class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"类别{class_id}"
            percentage = (count / total_objects) * 100 if total_objects > 0 else 0
            print(f"  {class_name} (ID: {class_id}): {count} 个对象 ({percentage:.1f}%)")
    else:
        print("未发现任何标注对象")

def main():
    """
    主函数：定义图像文件夹和标签文件夹路径
    """
    # 定义图像文件夹路径
    image_folder = r"C:\Users\Eugene\Desktop\vehicle_dataset\obb\20_png"
    
    # 定义标签文件夹路径
    label_folder = r"C:\Users\Eugene\Desktop\vehicle_dataset\obb\20_png"
    
    # 定义输出文件夹路径
    output_folder = label_folder + r"\vis"
    # 可选：定义类别名称（如果知道的话）
    class_names = None  # 例如: ["person", "car", "bicycle"]
    
    # 处理图像和标签
    process_images_and_labels(image_folder, label_folder, output_folder, class_names)

if __name__ == "__main__":
    main()
