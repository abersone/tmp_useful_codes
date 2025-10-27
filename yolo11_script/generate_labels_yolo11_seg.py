import ultralytics
from ultralytics import YOLO
import cv2
import os
import numpy as np
from pathlib import Path

def normalize_coordinates(points, img_width, img_height):
    """Normalize coordinates by image dimensions"""
    points_copy = points.copy()
    points_copy[:, 0] = points_copy[:, 0] / img_width
    points_copy[:, 1] = points_copy[:, 1] / img_height
    return points_copy

def denormalize_coordinates(points, img_width, img_height):
    """Denormalize coordinates to image dimensions"""
    points_copy = points.copy()
    points_copy[:, 0] = points_copy[:, 0] * img_width
    points_copy[:, 1] = points_copy[:, 1] * img_height
    return points_copy.astype(np.int32)

def process_images(model_path, image_folder, output_folder):
    """
    Process images with YOLO segmentation model and save results
    1. Load model and setup directories
    2. For each image:
        a. Run inference
        b. Save normalized coordinates to txt
        c. Visualize results from txt file
    """
    # 1. 初始化设置
    # 加载模型
    model = YOLO(model_path)
    
    # 创建输出文件夹
    output_folder = Path(output_folder)
    vis_folder = output_folder / 'visualizations_inference'
    txt_folder = output_folder / 'labels_inference'
    vis_folder.mkdir(parents=True, exist_ok=True)
    txt_folder.mkdir(parents=True, exist_ok=True)
    
    # 定义颜色映射
    colors = {
        0: (0, 255, 0),    # 绿色
        1: (255, 0, 255),  # 紫色
        2: (0, 255, 255)   # 黄色
    }
    
    # 获取所有图像文件
    image_files = []
    for ext in ('*.jpg', '*.png'):
        image_files.extend(Path(image_folder).glob(ext))
    
    # 2. 处理每张图像
    for img_path in image_files:
        print(f"Processing {img_path}")
        
        # 2.1 读取图像
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not read image: {img_path}")
            continue
        
        # 获取图像尺寸
        img_height, img_width = img.shape[:2]
        img_name = img_path.stem
        
        # 2.2 运行模型推理
        results = model.predict(source=img, save=False, save_txt=False)
        
        # 2.3 保存标签文件（归一化坐标）
        txt_path = txt_folder / f"{img_name}.txt"
        with open(txt_path, 'w') as f:
            for result in results:
                if result.masks is not None:
                    for seg_idx, mask in enumerate(result.masks.xy):
                        # 获取类别ID
                        class_id = int(result.boxes.cls[seg_idx])
                        # 转换为numpy数组并归一化
                        points = np.array(mask)
                        norm_points = normalize_coordinates(points, img_width, img_height)
                        # 转换为字符串并保存
                        points_str = ' '.join([f"{x:.6f} {y:.6f}" for x, y in norm_points])
                        f.write(f"{class_id} {points_str}\n")
        
        # 2.4 从txt文件读取并可视化
        vis_path = vis_folder / f"{img_name}_vis.jpg"
        verify_img = img.copy()
        
        if txt_path.exists():
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    if class_id not in colors:
                        continue
                    
                    # 转换字符串坐标为浮点数对
                    coords = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    coords = coords.reshape(-1, 2)
                    
                    # 反归一化坐标
                    points = denormalize_coordinates(coords, img_width, img_height)
                    
                    # 绘制多边形和标签
                    cv2.polylines(verify_img, [points], 
                                isClosed=True, 
                                color=colors[class_id], 
                                thickness=1)
                    
                    centroid = points.mean(axis=0).astype(int)
                    label = f"{class_id}"
                    cv2.putText(verify_img, 
                              label,
                              (centroid[0], centroid[1]),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.7,
                              colors[class_id],
                              2)
        
        # 2.5 保存可视化结果
        cv2.imwrite(str(vis_path), verify_img)
        print(f"Saved results for {img_path}")

if __name__ == "__main__":
    # Example usage
    model_path = "/home/liuqin/Desktop/projects/ultralytics/runs/segment/train/20250803_191424/seg_exp_s_512_20250803_191424/train/weights/best.pt"
    image_folder = "/home/liuqin/Desktop/projects/datasets/ball_lead_pad_dataset/20250812_ball/image"
    output_folder = "/home/liuqin/Desktop/projects/datasets/ball_lead_pad_dataset/20250812_ball"
    
    process_images(model_path, image_folder, output_folder)
