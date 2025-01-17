import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def calculate_white_ratio(img, coords, white_threshold=200):
    """计算目标框内白色区域的占比和像素数量
    
    Args:
        img: 输入图像
        coords: 归一化的坐标点 [x1,y1,x2,y2,x3,y3,x4,y4]
        white_threshold: 判定为白色的阈度值
    
    Returns:
        tuple: (白色区域占比, 白色像素数量)
    """
    h, w = img.shape[:2]
    
    # 将归一化坐标转换为实际坐标
    points = np.array([[float(coords[i]) * w, float(coords[i+1]) * h] 
                      for i in range(0, len(coords), 2)], dtype=np.int32)
    
    # 创建mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    
    # 获取目标区域
    roi = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[mask == 1]
    
    if len(roi) == 0:
        return 0.0, 0
    
    # 计算白色像素
    white_pixels = np.sum(roi >= white_threshold)
    total_pixels = len(roi)
    
    return white_pixels / total_pixels, white_pixels

def process_dataset(base_path, output_path, white_gray_threshold=200,white_ratio_threshold=0.5, white_pixels_threshold=1000):
    """处理数据集，修改符合条件的类别2标签为类别0
    
    Args:
        base_path: 数据集根目录
        output_path: 输出目录
        white_ratio_threshold: 白色区域占比阈值
        white_pixels_threshold: 白色像素数量阈值
    """
    base_path = Path(base_path)
    output_path = Path(output_path)
    
    # 创建输出目录
    for subset in ['train', 'val']:
        (output_path / 'images' / subset).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / subset).mkdir(parents=True, exist_ok=True)
    
    # 修改统计字典结构
    stats = {
        'train': {
            'total_cls2': 0,
            'converted': 0,
            'processed_images': 0
        },
        'val': {
            'total_cls2': 0,
            'converted': 0,
            'processed_images': 0
        }
    }
    
    # 处理训练集和验证集
    for subset in ['train', 'val']:
        print(f"\n处理{subset}集...")
        
        img_dir = base_path / 'images' / subset
        label_dir = base_path / 'labels' / subset
        
        # 获取所有图像文件
        img_files = list(img_dir.glob('*.jpg'))
        
        for img_path in tqdm(img_files):
            stats[subset]['processed_images'] += 1
            
            # 读取图像
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"无法读取图像: {img_path}")
                continue
            
            # 读取对应的标签文件
            label_path = label_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                print(f"警告：找不到对应的标签文件: {label_path}")
                continue
            
            # 读取标签内容
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            modified = False
            new_lines = []
            
            # 处理每个目标
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 9:
                    new_lines.append(line)
                    continue
                
                cls_id = int(float(parts[0]))
                if cls_id == 2:
                    stats[subset]['total_cls2'] += 1
                    coords = [float(x) for x in parts[1:]]
                    
                    # 计算白色区域占比和像素数量
                    white_ratio, white_pixels = calculate_white_ratio(img, coords, white_threshold=white_gray_threshold)
                    
                    # 如果白色区域占比超过阈值或白色像素数量超过阈值，修改类别
                    if white_ratio >= white_ratio_threshold or white_pixels >= white_pixels_threshold:
                        parts[0] = '0'
                        modified = True
                        stats[subset]['converted'] += 1
                
                new_lines.append(' '.join(parts) + '\n')
            
            # 复制图像到新目录
            cv2.imwrite(str(output_path / 'images' / subset / img_path.name), img)
            
            # 保存标签文件（无论是否修改）
            out_label_path = output_path / 'labels' / subset / label_path.name
            with open(out_label_path, 'w') as f:
                f.writelines(new_lines)
    
    # 打印分开的统计信息
    print("\n=== 处理统计 ===")
    for subset in ['train', 'val']:
        print(f"\n{subset}集统计:")
        print(f"处理图像总数: {stats[subset]['processed_images']}")
        print(f"类别2目标总数: {stats[subset]['total_cls2']}")
        print(f"转换为类别0的目标数: {stats[subset]['converted']}")
        if stats[subset]['total_cls2'] > 0:
            print(f"转换比例: {stats[subset]['converted']/stats[subset]['total_cls2']*100:.2f}%")
    
    # 打印总体统计
    # total_images = sum(stats[subset]['processed_images'] for subset in ['train', 'val'])
    # total_cls2 = sum(stats[subset]['total_cls2'] for subset in ['train', 'val'])
    # total_converted = sum(stats[subset]['converted'] for subset in ['train', 'val'])
    
    # print("\n总体统计:")
    # print(f"总处理图像数: {total_images}")
    # print(f"总类别2目标数: {total_cls2}")
    # print(f"总转换目标数: {total_converted}")
    # if total_cls2 > 0:
    #     print(f"总转换比例: {total_converted/total_cls2*100:.2f}%")

def main():
    # 设置路径和参数
    base_path = "/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/liuqin_data/test_dataset"  # 修改为你的数据集路径
    output_path = "/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/liuqin_data/test_dataset_transformed_cls2"      # 修改为你想要的输出路径
    white_ratio_threshold = 0.1           # 白色区域占比阈值
    white_pixels_threshold = 900         # 白色像素数量阈值
    white_gray_threshold = 180            # 白色灰度阈值

    # 处理数据集
    process_dataset(base_path, output_path, white_gray_threshold, white_ratio_threshold, white_pixels_threshold)

if __name__ == "__main__":
    main()
