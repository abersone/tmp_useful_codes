import os
import shutil
from pathlib import Path
from tqdm import tqdm

def sample_yolo_dataset(input_folder, output_folder, sample_count):
    """
    对YOLO格式的数据集进行顺序采样，当采样数量大于实际文件数量时复制所有文件
    
    Args:
        input_folder (str): 输入数据集的根目录
        output_folder (str): 输出数据集的根目录
        sample_count (dict): 采样数量，格式为{'train': n1, 'val': n2}
    """
    # 检查采样数量是否合法
    for split, count in sample_count.items():
        if count < 0:
            raise ValueError(f"{split}集的采样数量不能为负数: {count}")
    
    # 创建输出目录结构
    for split in ['train', 'val']:
        for folder in ['images', 'labels']:
            os.makedirs(os.path.join(output_folder, folder, split), exist_ok=True)
    
    # 对训练集和验证集分别进行采样
    for split in ['train', 'val']:
        # 获取图片文件列表
        image_dir = os.path.join(input_folder, 'images', split)
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        # 获取实际文件数量
        total_files = len(image_files)
        
        if sample_count[split] == 0:
            print(f'警告: {split}集的采样数量为0，将跳过该数据集的采样')
            continue
            
        # 确定采样数量
        sample_size = min(sample_count[split], total_files)
        # 获取指定数量的文件
        sampled_files = image_files[:sample_size]
        
        # 复制文件（添加进度条）
        for image_file in tqdm(sampled_files, desc=f'正在处理{split}集', unit='文件'):
            # 复制图片
            src_image = os.path.join(input_folder, 'images', split, image_file)
            dst_image = os.path.join(output_folder, 'images', split, image_file)
            shutil.copy2(src_image, dst_image)
            
            # 复制对应的标签文件
            label_file = Path(image_file).stem + '.txt'
            src_label = os.path.join(input_folder, 'labels', split, label_file)
            dst_label = os.path.join(output_folder, 'labels', split, label_file)
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
        
        print(f'{split}集采样完成: {len(sampled_files)}/{total_files}')

def main():
    # 配置参数
    input_folder = '/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/liuqin_data/test_dataset'  # 输入数据集路径
    output_folder = '/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/liuqin_data/test_dataset_sampling'  # 输出数据集路径
    sample_count = {
        'train': 2000,  # 训练集采样数量
        'val': 2000      # 验证集采样数量
    }
    
    # 执行采样
    sample_yolo_dataset(input_folder, output_folder, sample_count)

if __name__ == '__main__':
    main()
