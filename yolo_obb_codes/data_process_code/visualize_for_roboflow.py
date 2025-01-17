import os
import cv2
import numpy as np
from pathlib import Path
import yaml
import argparse
from tqdm import tqdm

def load_dataset_config(yaml_path):
    """加载数据集配置文件"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def visualize_rotated_labels(base_path, subset='train', test_mode=False):
    """可视化旋转框标签并检查异常情况
    
    Args:
        base_path: 数据集根目录路径
        subset: 'train' 或 'val'
        test_mode: 是否为测试模式，如果为True则只处理前10张图像
    """
    # 构建图像和标签目录路径
    img_dir = Path(base_path) / 'images' / subset
    label_dir = Path(base_path) / 'labels' / subset
    
    # 加载类别名称
    yaml_path = Path(base_path) / 'dataset.yaml'
    config = load_dataset_config(yaml_path)
    class_names = config['names']
    
    # 统计变量
    empty_labels = []
    missing_pairs = []
    total_images = 0
    
    # 为每个类别分配不同的颜色
    colors = [
        (255, 0, 0),    # 蓝色 - dust
        (216, 191, 216),    # 浅紫色 - glue
    ]
    
    # 创建可视化输出目录
    vis_dir = Path(base_path) / 'visualizations' / subset
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    img_files = list(img_dir.glob('*.jpg'))
    total_images = len(img_files)
    
    print(f"\n开始处理{subset}集...")
    print(f"找到{total_images}张图像")
    
    # 在测试模式下限制处理数量
    if test_mode and total_images > 10:
        img_files = img_files[:10]
        print("测试模式：只处理前10张图像")
    
    for img_path in tqdm(img_files):
        # 构建对应的标签文件路径
        label_path = label_dir / f"{img_path.stem}.txt"
        
        # 检查标签文件是否存在
        if not label_path.exists():
            missing_pairs.append(f"缺失标签文件: {label_path}")
            print(f"缺失标签文件: {label_path}")
            continue
        
        # 读取图像
        img = cv2.imread(str(img_path))
        if img is None:
            missing_pairs.append(f"无法读取图像: {img_path}")
            print(f"无法读取图像: {img_path}")
            continue
        # 将图像左右拼接
        img_crop = img.copy()        
        # 读取标签
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) == 0:
                empty_labels.append(str(label_path))
                print(f"发现空标签文件: {label_path}")
                continue
            
            corrected_lines = []
            needs_correction = False
            
            # 处理每个目标框
            for line in lines:
                parts = line.strip().split()
                
                # 检查是否是5个点的格式
                if len(parts) == 11:  # class + 10个坐标值（5个点）
                    needs_correction = True
                    # 删除最后一个点
                    parts = parts[:-2]
                    corrected_lines.append(' '.join(parts))
                    print(f"修正了标签格式: {label_path}")
                elif len(parts) == 9:  # 已经是正确格式
                    corrected_lines.append(line.strip())
                else:
                    print(f"标签格式异常 {label_path}: {line}")
                    continue
                
                # 解析类别和坐标（使用修正后的格式）
                cls_id = int(parts[0])
                points = np.array([
                    [float(parts[i]) * img.shape[1], float(parts[i+1]) * img.shape[0]] 
                    for i in range(1, 9, 2)
                ], dtype=np.int32)
                
                # 绘制旋转框
                cv2.polylines(img, [points], True, colors[cls_id], 2)
                
                # 添加类别标签
                label = class_names[cls_id]
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                
                # 找到框的最上方点作为标签位置
                text_x = min(points[:, 0])
                text_y = min(points[:, 1]) - 5
                
                # 计算文本大小
                (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                
                # 绘制文本背景
                cv2.rectangle(img, 
                            (int(text_x), int(text_y - text_h - 5)),
                            (int(text_x + text_w), int(text_y)),
                            colors[cls_id], -1)
                
                # 绘制文本
                cv2.putText(img, label,
                           (int(text_x), int(text_y)),
                           font, font_scale, (255, 255, 255), thickness)
            
            # 如果有任何需要修正的行，就保存整个文件
            if needs_correction:
                with open(label_path, 'w') as f:
                    for line in corrected_lines:
                        f.write(line + '\n')
                print(f"保存了修正后的标签文件: {label_path}")
            
            img_width = img.shape[1]
            img_height = img.shape[0]
            
            # 创建一个新的画布,宽度是原图的2倍
            combined_img = np.zeros((img_height, img_width * 2, 3), dtype=np.uint8)
            combined_img[:, :img_width] = img
            combined_img[:, img_width:] = img_crop

            # 保存可视化结果
            cv2.imwrite(str(vis_dir / img_path.name), combined_img)
            
        except Exception as e:
            print(f"处理文件时出错 {label_path}: {str(e)}")
    
    # 打印统计信息
    print("\n=== 统计信息 ===")
    print(f"总图像数量: {total_images}")
    print(f"空标签数量: {len(empty_labels)}")
    print(f"异常配对数量: {len(missing_pairs)}")
    
    # 打印详细信息
    if empty_labels:
        print("\n空标签文件:")
        for path in empty_labels:
            print(f"  - {path}")
    
    if missing_pairs:
        print("\n异常配对:")
        for msg in missing_pairs:
            print(f"  - {msg}")

if __name__ == "__main__":
    # 设置数据集根目录
    base_path = "/mnt/nfs/AOI_detection/ccm/data/v5/relabel_data/liuqin_dataset/yolo_obb_contrast42_area30_augment5058"
    
    # 处理训练集和验证集
    for subset in ['train', 'val']:
        visualize_rotated_labels(base_path, subset, test_mode=False)