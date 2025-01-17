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
    根据四个顶点坐标获取最小外接矩形并裁剪图像，可指定扩展边界
    Args:
        image: 输入图像
        points: 四个顶点坐标，numpy数组，形状为(4, 2)
        delta: 向外扩展的像素数
    Returns:
        cropped: 裁剪后的图像
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
    return cropped

def process_images(img_folder, label_folder, target_folder, target_cls, delta=10):
    """
    处理图像文件夹
    """
    # 创建目标文件夹
    os.makedirs(target_folder, exist_ok=True)
    
    # 遍历所有图像
    # 获取所有图片路径
    img_paths = list(Path(img_folder).glob('*.jpg'))
    total = len(img_paths)
    
    # 使用tqdm创建进度条
    for i, img_path in enumerate(tqdm(img_paths, desc="处理图片")):
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
                cropped = crop_rotated_box(image, points, delta=delta)
                if cropped is not None:
                    # 保存裁剪后的图像
                    save_path = os.path.join(target_folder, f"{img_path.stem}_{cls_count}.jpg")
                    cv2.imwrite(save_path, cropped)
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
    
    # 处理图像
    process_images(img_folder, label_folder, target_folder, target_cls, delta=delta)