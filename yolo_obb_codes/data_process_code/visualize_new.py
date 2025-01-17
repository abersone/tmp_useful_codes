import os
import cv2
import numpy as np
from pathlib import Path
import yaml
import argparse
from tqdm import tqdm

def load_dataset_config(yaml_path):
    """加载数据集配置文件"""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"加载配置文件失败: {str(e)}")
        return None

def generate_colors(num_classes):
    """为每个类别生成唯一的颜色"""
    np.random.seed(42)  # 固定随机种子，确保颜色一致性
    colors = []
    for i in range(num_classes):
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        colors.append(color)
    return colors

def check_dataset_structure(base_path):
    """检查数据集目录结构"""
    required_dirs = [
        'images/train', 'images/val',
        'labels/train', 'labels/val'
    ]
    
    for dir_path in required_dirs:
        full_path = Path(base_path) / dir_path
        if not full_path.exists():
            raise ValueError(f"目录不存在: {full_path}")

def visualize_rotated_labels(base_path, subset='train', test_mode=False, save_vis=True):
    """可视化旋转框标签并进行统计分析
    
    Args:
        base_path: 数据集根目录路径
        subset: 'train' 或 'val'
        test_mode: 是否为测试模式
        save_vis: 是否保存可视化结果
    """
    # 构建路径
    img_dir = Path(base_path) / 'images' / subset
    label_dir = Path(base_path) / 'labels' / subset
    
    # 加载类别配置
    yaml_path = Path(base_path) / 'dataset.yaml'
    config = load_dataset_config(yaml_path)
    if config is None or 'names' not in config:
        print("警告：未找到类别配置，使用默认类别名称")
        class_names = [f"class_{i}" for i in range(10)]  # 默认10个类别
    else:
        class_names = config['names']
    
    # 初始化统计变量
    stats = {
        'empty_labels': [],
        'missing_pairs': [],
        'class_counts': {},
        'orphaned_images': [],
        'orphaned_labels': [],
        'invalid_labels': [],
        'total_images': 0,
        'total_objects': 0
    }
    
    # 生成颜色映射
    colors = generate_colors(len(class_names))
    
    # 获取所有文件
    img_files = set(f for f in img_dir.glob('*.jpg'))
    label_files = set(f for f in label_dir.glob('*.txt'))
    
    # 检查文件配对
    for img_file in img_files:
        label_file = label_dir / f"{img_file.stem}.txt"
        if not label_file.exists():
            stats['orphaned_images'].append(img_file)
    
    for label_file in label_files:
        img_file = img_dir / f"{label_file.stem}.jpg"
        if not img_file.exists():
            stats['orphaned_labels'].append(label_file)
    
    # 获取有效的图像文件
    valid_img_files = [f for f in img_files if (label_dir / f"{f.stem}.txt").exists()]
    stats['total_images'] = len(valid_img_files)
    
    if test_mode:
        valid_img_files = valid_img_files[:min(10, len(valid_img_files))]
        print(f"测试模式：处理前{len(valid_img_files)}张图像")
    
    # 创建可视化输出目录
    if save_vis:
        vis_dir = Path(base_path) / 'visualizations' / subset
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理每个图像
    for img_path in tqdm(valid_img_files, desc=f"处理{subset}集"):
        img = cv2.imread(str(img_path))
        if img is None:
            stats['missing_pairs'].append(f"无法读取图像: {img_path}")
            continue
        
        label_path = label_dir / f"{img_path.stem}.txt"
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) == 0:
                stats['empty_labels'].append(str(label_path))
                continue
            
            vis_img = img.copy()
            
            for line in lines:
                parts = line.strip().split()
                
                if len(parts) != 9:
                    stats['invalid_labels'].append(
                        f"{label_path}: 行 '{line.strip()}' 格式错误"
                    )
                    continue
                
                try:
                    # 解析类别和坐标
                    cls_id = int(float(parts[0]))
                    if cls_id >= len(class_names):
                        stats['invalid_labels'].append(
                            f"{label_path}: 类别ID {cls_id} 超出范围"
                        )
                        continue
                    
                    # 更新类别统计
                    stats['class_counts'][cls_id] = \
                        stats['class_counts'].get(cls_id, 0) + 1
                    stats['total_objects'] += 1
                    
                    # 解析坐标点
                    points = np.array([
                        [float(parts[i]) * img.shape[1], 
                         float(parts[i+1]) * img.shape[0]]
                        for i in range(1, 9, 2)
                    ], dtype=np.int32)
                    
                    # 绘制旋转框
                    cv2.polylines(vis_img, [points], True, colors[cls_id], 2)
                    
                    # 添加类别标签
                    label = class_names[cls_id]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    
                    # 找到框的最上方点作为标签位置
                    text_x = int(min(points[:, 0]))
                    text_y = int(min(points[:, 1])) - 5
                    
                    # 计算文本大小
                    (text_w, text_h), _ = cv2.getTextSize(
                        label, font, font_scale, thickness
                    )
                    
                    # 绘制文本背景和文本
                    cv2.rectangle(
                        vis_img,
                        (text_x, text_y - text_h - 5),
                        (text_x + text_w, text_y),
                        colors[cls_id], -1
                    )
                    cv2.putText(
                        vis_img, label,
                        (text_x, text_y),
                        font, font_scale, (255, 255, 255), thickness
                    )
                    
                except (ValueError, IndexError) as e:
                    stats['invalid_labels'].append(
                        f"{label_path}: 解析错误 - {str(e)}"
                    )
                    continue
            
            if save_vis:
                cv2.imwrite(str(vis_dir / img_path.name), vis_img)
                
        except Exception as e:
            print(f"处理文件出错 {label_path}: {str(e)}")
    
    return stats

def print_statistics(stats, subset):
    """打印统计信息"""
    print(f"\n=== {subset}集统计信息 ===")
    print(f"总图像数量: {stats['total_images']}")
    print(f"总目标数量: {stats['total_objects']}")
    
    print("\n=== 类别统计 ===")
    for cls_id, count in sorted(stats['class_counts'].items()):
        print(f"类别 {cls_id}: {count} 个目标")
    
    print("\n=== 异常统计 ===")
    print(f"空标签文件数量: {len(stats['empty_labels'])}")
    print(f"无效标签数量: {len(stats['invalid_labels'])}")
    print(f"孤立图像文件数量: {len(stats['orphaned_images'])}")
    print(f"孤立标签文件数量: {len(stats['orphaned_labels'])}")
    
    # 打印详细信息
    if stats['empty_labels']:
        print("\n空标签文件:")
        for path in stats['empty_labels']:
            print(f"  - {path}")
    
    if stats['invalid_labels']:
        print("\n无效标签:")
        for msg in stats['invalid_labels']:
            print(f"  - {msg}")
    
    if stats['orphaned_images']:
        print("\n孤立图像文件:")
        for path in stats['orphaned_images']:
            print(f"  - {path}")
    
    if stats['orphaned_labels']:
        print("\n孤立标签文件:")
        for path in stats['orphaned_labels']:
            print(f"  - {path}")

def main():
    # 直接设置参数
    data_path = "/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/liuqin_data/test_dataset_transformed_cls2"  # 修改为你的数据集路径
    test_mode = False  # 是否使用测试模式（只处理少量图像）
    save_vis = True   # 是否保存可视化结果
    
    try:
        # 检查数据集结构
        check_dataset_structure(data_path)
        
        # 处理训练集和验证集
        for subset in ['train', 'val']:
            print(f"\n处理{subset}集...")
            stats = visualize_rotated_labels(
                data_path,
                subset,
                test_mode=test_mode,
                save_vis=save_vis
            )
            print_statistics(stats, subset)
            
    except Exception as e:
        print(f"程序执行出错: {str(e)}")

if __name__ == "__main__":
    main()