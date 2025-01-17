import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def parse_rotate_label(label_path, img_width, img_height):
    """
    解析标注文件，格式为：cls x1 y1 x2 y2 x3 y3 x4 y4
    返回: [(cls, points), ...]，其中points为numpy数组，形状为(4, 2)
    """
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            if len(data) == 9:  # cls x1 y1 x2 y2 x3 y3 x4 y4
                cls = int(float(data[0]))
                # 转换归一化坐标到实际坐标
                points = np.array([
                    [float(data[1]) * img_width, float(data[2]) * img_height],
                    [float(data[3]) * img_width, float(data[4]) * img_height],
                    [float(data[5]) * img_width, float(data[6]) * img_height],
                    [float(data[7]) * img_width, float(data[8]) * img_height]
                ], dtype=np.float32)
                boxes.append((cls, points))
    return boxes

def crop_rotated_box(image, points, delta=0):
    """
    根据四个顶点坐标获取最小外接矩形并裁剪图像，同时返回裁剪区域的坐标
    Args:
        image: 输入图像
        points: 四个顶点坐标，numpy数组，形状为(4, 2)
        delta: 向外扩展的像素数
    Returns:
        cropped: 裁剪后的图像
        (x_min, y_min): 裁剪区域左上角坐标
    """
    # 获取最小外接矩形
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    
    # 获取最小外接矩形的坐标范围，并扩展delta像素
    x_coords = box[:, 0]
    y_coords = box[:, 1]
    x_min = max(0, np.min(x_coords) - delta)
    x_max = min(image.shape[1], np.max(x_coords) + delta)
    y_min = max(0, np.min(y_coords) - delta)
    y_max = min(image.shape[0], np.max(y_coords) + delta)
    
    # 裁剪图像
    cropped = image[y_min:y_max, x_min:x_max]
    return cropped, (x_min, y_min)

def convert_points_to_cropped(points, x_min, y_min, crop_width, crop_height):
    """
    将原始图像中的点坐标转换为裁剪图像中的归一化坐标
    """
    # 平移坐标
    points_cropped = points.copy()
    points_cropped[:, 0] -= x_min
    points_cropped[:, 1] -= y_min
    
    # 归一化坐标
    points_cropped[:, 0] /= crop_width
    points_cropped[:, 1] /= crop_height
    
    return points_cropped

def visualize_label(image_path, label_path):
    """
    可视化标签
    """
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    boxes = parse_rotate_label(label_path, w, h)
    
    vis_img = image.copy()
    for cls, points in boxes:
        # 转换回像素坐标
        points = points.astype(np.int32)
        # 绘制旋转框
        cv2.polylines(vis_img, [points], True, (0, 255, 0), 2)
        # 在左上角标注类别
        cv2.putText(vis_img, f'cls:{cls}', tuple(points[0]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return vis_img

def process_images(img_folder, label_folder, target_folder, target_cls, delta=10, debug=False):
    """
    处理图像文件夹
    Args:
        debug: 是否为调试模式，若为True，则只处理前20张图片
    """
    # 创建目标文件夹
    os.makedirs(target_folder, exist_ok=True)
    os.makedirs(os.path.join(target_folder, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(target_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(target_folder, 'visualization'), exist_ok=True)
    
    # 获取所有图片路径
    img_paths = list(Path(img_folder).glob('*.jpg'))
    if debug:
        img_paths = img_paths[:20]  # debug模式下只处理前20张图片
        print(f"Debug模式：仅处理前20张图片")
    
    # 使用tqdm创建进度条
    for img_path in tqdm(img_paths, desc="处理图片"):
        # 获取对应的label文件
        label_path = Path(label_folder) / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
            
        # 读取图像
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"无法读取图像: {img_path}")
            continue
            
        h, w = image.shape[:2]
        
        # 解析标签
        boxes = parse_rotate_label(str(label_path), w, h)
        
        # 处理指定类别的框
        cls_count = 0
        for cls, points in boxes:
            if cls == target_cls:
                # 裁剪图像
                cropped, (x_min, y_min) = crop_rotated_box(image, points, delta=delta)
                if cropped is not None:
                    crop_h, crop_w = cropped.shape[:2]
                    
                    # 转换坐标
                    points_cropped = convert_points_to_cropped(
                        points, x_min, y_min, crop_w, crop_h)
                    
                    # 保存裁剪后的图像
                    save_name = f"{img_path.stem}_{cls_count}"
                    image_save_path = os.path.join(target_folder, 'images', f"{save_name}.jpg")
                    cv2.imwrite(image_save_path, cropped)
                    
                    # 保存转换后的标签
                    label_save_path = os.path.join(target_folder, 'labels', f"{save_name}.txt")
                    with open(label_save_path, 'w') as f:
                        coords = points_cropped.flatten()
                        label_line = f"{cls} " + " ".join([f"{x:.6f}" for x in coords])
                        f.write(label_line)
                    
                    # 可视化验证
                    vis_img = visualize_label(image_save_path, label_save_path)
                    vis_save_path = os.path.join(target_folder, 'visualization', f"{save_name}.jpg")
                    cv2.imwrite(vis_save_path, vis_img)
                    
                    cls_count += 1
                else:
                    print(f"裁剪失败: {img_path}, box {cls_count}")

if __name__ == "__main__":
    # 设置路径和参数
    img_folder = "/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/train_data/after_check/train_data_all_yolo/images/train"
    label_folder = "/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/train_data/after_check/train_data_all_yolo/labels/train"
    target_folder = "/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/liuqin_data/dirt_tmpls"
    target_cls = 2  # 目标类别
    delta = 40
    debug = False  # 设置debug模式
    
    # 处理图像
    process_images(img_folder, label_folder, target_folder, target_cls, delta=delta, debug=debug)