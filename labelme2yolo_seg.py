#!/usr/bin/env python3
"""
Convert LabelMe JSON annotations to YOLO-SEG format

This script converts LabelMe JSON annotations (rectangle and polygon) to YOLO-SEG format.
For rectangles, it converts them to 4-point polygons before conversion.

Usage:
    python labelme2yoloseg.py --input /path/to/labelme/json/files --output /path/to/yolo/seg/dataset
    python labelme2yoloseg.py --input /path/to/labelme/json/files --output /path/to/yolo/seg/dataset --test
    python labelme2yoloseg.py --input /path/to/labelme/json/files --output /path/to/yolo/seg/dataset --train-ratio 0.7 --classes ball lead wire pad
"""

import os
import json
import shutil
from pathlib import Path
import random
import cv2
import numpy as np
from tqdm import tqdm
import yaml
import time
import argparse

def convert_rectangle_to_polygon(points):
    """
    Convert rectangle points (2 points) to polygon points (4 points)
    
    Args:
        points: List of 2 points [[x1,y1], [x2,y2]]
        
    Returns:
        List of 4 points [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
    """
    if len(points) != 2:
        raise ValueError("Rectangle must have exactly 2 points")
    
    x1, y1 = points[0]
    x2, y2 = points[1]
    
    return [
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ]

def normalize_points(points, image_width, image_height):
    """
    Normalize points to [0,1] range
    
    Args:
        points: List of [x,y] points
        image_width: Image width
        image_height: Image height
        
    Returns:
        List of normalized points
    """
    return [[x / image_width, y / image_height] for x, y in points]

def convert_labelme_to_yolo_seg(json_file, class_names):
    """
    Convert LabelMe JSON annotation to YOLO-SEG format
    
    Args:
        json_file: Path to JSON file
        class_names: List of class names
        
    Returns:
        List of YOLO-SEG format annotations
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    
    image_width = data["imageWidth"]
    image_height = data["imageHeight"]
    
    yolo_annotations = []
    for shape in data["shapes"]:
        label = shape["label"]
        if label not in class_names:
            continue
        
        class_id = class_names.index(label)
        points = shape["points"]
        
        # Convert rectangle to polygon if needed
        if shape["shape_type"] == "rectangle":
            points = convert_rectangle_to_polygon(points)
        
        # Normalize points
        normalized_points = normalize_points(points, image_width, image_height)
        
        # Flatten points and create YOLO-SEG format annotation
        flattened_points = [coord for point in normalized_points for coord in point]
        yolo_annotation = f"{class_id} " + " ".join([f"{x:.6f}" for x in flattened_points])
        yolo_annotations.append(yolo_annotation)
    
    return yolo_annotations

def find_image_file(json_file):
    """Find corresponding image file for a JSON file"""
    for ext in [".jpg", ".JPG", ".jpeg", ".png", ".bmp"]:
        img_file = json_file.with_suffix(ext)
        if img_file.exists():
            return img_file
    return None

def process_dataset(labelme_path, yolo_path, class_names, test_mode=False):
    """
    Process the entire dataset
    
    Args:
        labelme_path: Path to LabelMe JSON files
        yolo_path: Path to save YOLO-SEG dataset
        class_names: List of class names
        test_mode: If True, process only a few files for testing
    """
    images_path = Path(yolo_path) / "images"
    labels_path = Path(yolo_path) / "labels"
    vis_path = Path(yolo_path) / "visualizations"
    
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)
    os.makedirs(vis_path, exist_ok=True)
    
    # 递归查找所有JSON文件
    json_files = []
    for json_file in Path(labelme_path).rglob("*.json"):
        json_files.append(json_file)
    
    if test_mode:
        json_files = random.sample(json_files, min(5, len(json_files)))
    
    processed_count = 0
    skipped_count = 0
    
    for json_file in tqdm(json_files, desc="Processing files"):
        try:
            # 查找对应的图像文件
            image_file = find_image_file(json_file)
            if not image_file:
                print(f"未找到图片文件: {json_file}")
                skipped_count += 1
                continue
            
            # 转换标注
            yolo_annotations = convert_labelme_to_yolo_seg(json_file, class_names)
            
            # 如果没有有效的标注，跳过此文件
            if not yolo_annotations:
                print(f"没有有效的标注: {json_file}")
                skipped_count += 1
                continue
            
            # 生成输出文件名（保持原始文件名，不添加前缀）
            base_name = json_file.stem
            label_file = labels_path / f"{base_name}.txt"
            output_image = images_path / image_file.name
            
            # 保存YOLO11格式的分割标签
            with open(label_file, "w") as f:
                f.write("\n".join(yolo_annotations))
            
            # 复制图像文件
            shutil.copy(image_file, output_image)
            
            # 生成可视化结果
            vis_output = vis_path / f"vis_{image_file.name}"
            visualize_yolo_seg_annotations(
                image_file,
                label_file,
                class_names,
                vis_output
            )
            
            processed_count += 1
            
        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {e}")
            skipped_count += 1
            continue
    
    print(f"处理完成！成功处理 {processed_count} 个文件，跳过 {skipped_count} 个文件")

def visualize_yolo_seg_annotations(image_path, annotation_path, class_names, output_path):
    """
    Visualize YOLO-SEG annotations
    
    Args:
        image_path: Path to image file
        annotation_path: Path to YOLO-SEG annotation file
        class_names: List of class names
        output_path: Path to save visualization
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"警告：无法读取图像 {image_path}")
        return
    
    height, width = image.shape[:2]
    
    # Define colors for different classes
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Pink
        (0, 255, 255),  # Yellow
    ]
    
    # Read annotations
    with open(annotation_path, "r") as f:
        annotations = f.readlines()
    
    # Draw each annotation
    for ann in annotations:
        parts = ann.strip().split()
        class_id = int(parts[0])
        points = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
        
        # Denormalize coordinates
        points[:, 0] *= width
        points[:, 1] *= height
        points = points.astype(np.int32)
        
        # Draw polygon
        color = colors[class_id % len(colors)]
        cv2.polylines(image, [points], True, color, 2)
        
        # Add class label
        label = class_names[class_id]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        
        # Calculate text background position
        text_x = points[0][0]
        text_y = points[0][1] - 5
        
        # Draw text background
        cv2.rectangle(image, 
                     (text_x, text_y - text_size[1] - 5),
                     (text_x + text_size[0], text_y),
                     color, -1)
        
        # Draw text
        cv2.putText(image, label,
                    (text_x, text_y),
                    font, font_scale, (255, 255, 255), thickness)
    
    # Save result
    cv2.imwrite(str(output_path), image)

def split_dataset(yolo_path, train_ratio=0.8):
    """
    Split dataset into train and validation sets
    
    Args:
        yolo_path: Path to YOLO-SEG dataset
        train_ratio: Ratio of training data
    """
    images_path = Path(yolo_path) / "images"
    labels_path = Path(yolo_path) / "labels"
    
    all_files = list(labels_path.glob("*.txt"))
    if not all_files:
        print("警告：没有找到标签文件")
        return
    
    random.seed(42)
    random.shuffle(all_files)
    
    split_index = int(len(all_files) * train_ratio)
    train_files = all_files[:split_index]
    val_files = all_files[split_index:]
    
    print(f"训练集: {len(train_files)} 个文件")
    print(f"验证集: {len(val_files)} 个文件")
    
    for split, files in [("train", train_files), ("val", val_files)]:
        split_images_path = images_path / split
        split_labels_path = labels_path / split
        
        os.makedirs(split_images_path, exist_ok=True)
        os.makedirs(split_labels_path, exist_ok=True)
        
        for file in tqdm(files, desc=f"Moving {split} files"):
            # 移动标签文件
            shutil.move(file, split_labels_path / file.name)
            
            # 查找并移动对应的图像文件
            base_name = file.stem
            image_found = False
            
            # 尝试不同的图像格式
            for ext in [".jpg", ".JPG", ".jpeg", ".png", ".bmp", ".BMP"]:
                image_file = images_path / f"{base_name}{ext}"
                if image_file.exists():
                    shutil.move(image_file, split_images_path / image_file.name)
                    image_found = True
                    break
            
            if not image_found:
                print(f"警告：未找到标签文件 {file.name} 对应的图像文件")

def generate_dataset_yaml(yolo_path, class_names):
    """
    Generate dataset.yaml file
    
    Args:
        yolo_path: Path to YOLO-SEG dataset
        class_names: List of class names
    """
    yaml_content = {
        'path': str(yolo_path),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    yaml_path = Path(yolo_path) / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Dataset YAML file generated: {yaml_path}")

def test_conversion(labelme_path, test_output_path, class_names):
    """
    Test the conversion process with a few files
    
    Args:
        labelme_path: Path to LabelMe JSON files
        test_output_path: Path to save test results
        class_names: List of class names
    """
    yolo_seg_test_path = Path(test_output_path) / "yolo_seg_data"
    visualization_path = Path(test_output_path) / "seg_visualizations"
    
    os.makedirs(yolo_seg_test_path, exist_ok=True)
    os.makedirs(visualization_path, exist_ok=True)
    
    process_dataset(labelme_path, yolo_seg_test_path, class_names, test_mode=True)
    
    for annotation_file in tqdm(list(Path(yolo_seg_test_path / "labels").glob("*.txt")), desc="Generating visualizations"):
        base_name = annotation_file.stem
        image_found = False
        
        # 尝试不同的图像格式
        for ext in [".jpg", ".JPG", ".jpeg", ".png", ".bmp", ".BMP"]:
            image_file = yolo_seg_test_path / "images" / f"{base_name}{ext}"
            if image_file.exists():
                output_image = visualization_path / f"visualized_seg_{image_file.name}"
                visualize_yolo_seg_annotations(image_file, annotation_file, class_names, output_image)
                image_found = True
                break
        
        if not image_found:
            print(f"警告：未找到标签文件 {annotation_file.name} 对应的图像文件")
    
    print(f"YOLO-SEG test data saved to {yolo_seg_test_path}")
    print(f"Segmentation visualization results saved to {visualization_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert LabelMe JSON annotations to YOLO-SEG format')
    parser.add_argument('--input', type=str, default='/data/golden_wire/dataset/final_organized',
                       help='Path to input directory containing LabelMe JSON files')
    parser.add_argument('--output', type=str, default='/data/golden_wire/dataset/yolo_seg_v9',
                       help='Path to output directory for YOLO-SEG dataset')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (process only a few files)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Ratio of training data (default: 0.8)')
    parser.add_argument('--classes', nargs='+', default=["ball", "lead", "wire", "pad"],
                       help='List of class names')
    parser.add_argument('--test-output', type=str, default='/data/golden_wire/dataset/yolo_seg_v9_test',
                       help='Path to save test results (only used in test mode)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input):
        raise ValueError(f"Input directory does not exist: {args.input}")
    
    if args.train_ratio <= 0 or args.train_ratio >= 1:
        raise ValueError("Train ratio must be between 0 and 1")
    
    if not args.classes:
        raise ValueError("At least one class must be specified")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    if args.test:
        test_conversion(args.input, args.test_output, args.classes)
    else:
        process_dataset(args.input, args.output, args.classes, test_mode=False)
        split_dataset(args.output, args.train_ratio)
        generate_dataset_yaml(args.output, args.classes)
        print("YOLO-SEG dataset conversion, splitting, and YAML generation completed!")

if __name__ == "__main__":
    main() 