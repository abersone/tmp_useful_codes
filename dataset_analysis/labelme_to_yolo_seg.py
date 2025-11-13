import json
import os
from pathlib import Path
from collections import OrderedDict


def create_class_mapping(json_folder):
    """
    扫描所有JSON文件，创建类别名称到ID的映射
    """
    class_to_id = OrderedDict()
    
    json_files = list(Path(json_folder).glob('*.json'))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 遍历所有shapes
            if 'shapes' in data:
                for shape in data['shapes']:
                    label = shape.get('label', '').strip()
                    if label and label not in class_to_id:
                        class_to_id[label] = len(class_to_id)
                        
        except Exception as e:
            print(f"读取文件 {json_file} 时出错: {e}")
            continue
    
    return class_to_id


def convert_labelme_to_yolo(json_file_path, class_mapping, img_width=0, img_height=0):
    """
    将单个labelme JSON文件转换为YOLO11分割格式
    
    Args:
        json_file_path: JSON文件路径
        class_mapping: 类别名称到ID的映射字典
        img_width: 图像宽度（如果JSON中没有，则使用此值）
        img_height: 图像高度（如果JSON中没有，则使用此值）
    
    Returns:
        YOLO格式的字符串列表，每行一个实例
    """
    yolo_lines = []
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取图像尺寸（优先使用JSON中的尺寸）
        if 'imageWidth' in data and 'imageHeight' in data:
            img_width = int(data['imageWidth'])
            img_height = int(data['imageHeight'])
        
        # 如果还是没有尺寸，报错
        if img_width == 0 or img_height == 0:
            raise ValueError(f"无法获取图像尺寸，请确保JSON文件中包含imageWidth和imageHeight，或者图像文件可访问")
        
        # 处理每个shape（多边形）
        if 'shapes' in data:
            for shape in data['shapes']:
                shape_type = shape.get('shape_type', '')
                label = shape.get('label', '').strip()
                points = shape.get('points', [])
                
                # 只处理多边形
                if shape_type != 'polygon' and shape_type != '':
                    print(f"警告: 跳过非多边形类型 {shape_type} 在文件 {json_file_path.name}")
                    continue
                
                if not label:
                    print(f"警告: 跳过无标签的shape在文件 {json_file_path.name}")
                    continue
                
                if len(points) < 3:
                    print(f"警告: 跳过点数少于3的多边形在文件 {json_file_path.name}")
                    continue
                
                # 获取类别ID
                if label not in class_mapping:
                    print(f"警告: 未知类别 '{label}' 在文件 {json_file_path.name}，跳过")
                    continue
                
                class_id = class_mapping[label]
                
                # 归一化坐标
                normalized_points = []
                for point in points:
                    x, y = point[0], point[1]
                    x_norm = x / img_width
                    y_norm = y / img_height
                    
                    # 限制坐标范围在[0, 1]
                    x_norm = max(0.0, min(1.0, x_norm))
                    y_norm = max(0.0, min(1.0, y_norm))
                    
                    normalized_points.extend([x_norm, y_norm])
                
                # 构建YOLO格式行：class_id x1 y1 x2 y2 ...
                yolo_line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_points])
                yolo_lines.append(yolo_line)
        
    except Exception as e:
        print(f"转换文件 {json_file_path} 时出错: {e}")
        return []
    
    return yolo_lines


def process_folder(input_folder, output_folder, class_mapping=None):
    """
    批量处理文件夹中的所有labelme JSON文件
    
    Args:
        input_folder: 包含JSON文件的输入文件夹
        output_folder: 输出TXT文件的文件夹
        class_mapping: 可选的类别映射字典，如果为None则自动创建
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # 检查输入文件夹
    if not input_path.exists():
        print(f"错误: 输入文件夹不存在: {input_folder}")
        return
    
    # 创建输出文件夹
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有JSON文件
    json_files = list(input_path.glob('*.json'))
    
    if not json_files:
        print(f"警告: 在 {input_folder} 中未找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    # 如果没有提供类别映射，自动创建
    if class_mapping is None:
        print("正在扫描JSON文件以创建类别映射...")
        class_mapping = create_class_mapping(input_folder)
        
        if not class_mapping:
            print("错误: 未能从JSON文件中找到任何类别标签")
            return
        
        print(f"发现 {len(class_mapping)} 个类别:")
        for label, class_id in class_mapping.items():
            print(f"  {label} -> {class_id}")
    
    # 处理每个JSON文件
    processed_count = 0
    skipped_count = 0
    
    # 导入cv2（如果需要从图像文件读取尺寸）
    try:
        import cv2
        cv2_available = True
    except ImportError:
        cv2_available = False
        print("警告: 未安装cv2，无法从图像文件读取尺寸，将仅使用JSON中的尺寸信息")
    
    for json_file in json_files:
        # 构建输出TXT文件路径
        txt_file = output_path / f"{json_file.stem}.txt"
        
        # 尝试读取图像尺寸（备用方案：如果JSON中没有尺寸，尝试从图像文件读取）
        img_width = 0
        img_height = 0
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 如果JSON中没有尺寸信息，尝试从图像文件读取
            if 'imageWidth' not in data or 'imageHeight' not in data:
                if 'imagePath' in data and cv2_available:
                    image_path = input_path.parent / data['imagePath']
                    if not image_path.exists():
                        # 尝试在同一文件夹中查找
                        image_path = input_path / Path(data['imagePath']).name
                    
                    if image_path.exists():
                        img = cv2.imread(str(image_path))
                        if img is not None:
                            img_height, img_width = img.shape[:2]
        except Exception as e:
            print(f"读取图像尺寸时出错 {json_file.name}: {e}")
        
        # 转换JSON到YOLO格式（函数内部会再次检查JSON中的尺寸）
        yolo_lines = convert_labelme_to_yolo(json_file, class_mapping, img_width, img_height)
        
        if yolo_lines:
            # 保存TXT文件
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_lines))
            processed_count += 1
            
            if processed_count % 50 == 0:
                print(f"已处理 {processed_count} 个文件...")
        else:
            skipped_count += 1
            # 创建空文件或跳过
            # txt_file.write_text('')  # 可选：创建空文件
    
    print(f"\n处理完成！")
    print(f"成功转换: {processed_count} 个文件")
    if skipped_count > 0:
        print(f"跳过: {skipped_count} 个文件")
    print(f"输出文件夹: {output_folder}")


def main():
    """
    主函数：定义输入和输出文件夹路径
    """
    # 定义输入文件夹（包含labelme JSON文件）
    input_folder = r"C:\Users\Eugene\Desktop\ref_json"
    
    # 定义输出文件夹（保存YOLO11格式的TXT文件）
    output_folder = r"C:\Users\Eugene\Desktop\ref_yolo"
    
    # 可选：手动定义类别映射（如果知道所有类别）
    class_mapping = {
        "ball": 0,
        "pad": 1,
        "wire": 2,
        "lead": 3
    }
    # 如果设置为None，则自动从JSON文件中扫描创建
    # class_mapping = None
    
    # 处理文件夹
    process_folder(input_folder, output_folder, class_mapping)


if __name__ == "__main__":
    main()

