import os
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

def robust_imread(image_file):
    try:
        path = str(Path(image_file).resolve())
        with open(path, 'rb') as f:
            img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f'robust_imread 读取失败: {image_file}, 错误: {e}')
        return None

def convert_yolo_obb_to_labelme(txt_file, image_file, class_names):
    # 读取图像以获取尺寸
    img = robust_imread(image_file)
    if img is None:
        print(f'无法读取图片: {image_file}')
        return None
    image_height, image_width = img.shape[:2]
    
    # 准备LabelMe格式的数据结构
    labelme_data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [],
        "imagePath": image_file.name,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width
    }
    
    # 读取YOLO OBB格式的标注
    with open(txt_file, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        # 获取坐标值
        coords = []
        for v in parts[1:9]:  # 只取8个坐标
            v = v.strip('[],')
            if v:
                coords.append(float(v))
        
        # 将坐标转换为点的形式 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        points = np.array(coords, dtype=np.float32).reshape(-1, 2)
        
        # # 反归一化坐标（x乘以宽度，y乘以高度）
        points[:, 0] = points[:, 0] * image_width
        points[:, 1] = points[:, 1] * image_height
        
        # 创建LabelMe格式的shape
        shape = {
            "label": class_names[class_id],
            "points": points.tolist(),  # 确保转换为Python列表
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        labelme_data["shapes"].append(shape)
    
    return labelme_data

def process_dataset(yolo_obb_path, labelme_path, class_names):
    yolo_obb_path = Path(yolo_obb_path)
    labelme_path = Path(labelme_path)
    labelme_path.mkdir(parents=True, exist_ok=True)
    
    txt_files = list(yolo_obb_path.glob("*.txt"))
    
    for txt_file in tqdm(txt_files, desc="Converting files"):
        found = False
        for suffix in [".bmp",".png",".jpg"]:
            image_file = txt_file.with_suffix(suffix)	
            image_file = Path(str(image_file).replace("org_imgs_labels_v6.6", "org_imgs"))
            if image_file.exists():
                found = True
                break

        if not found:
            print(f"Warning: Image file not found for {image_file}")
            continue
        
        labelme_data = convert_yolo_obb_to_labelme(txt_file, image_file, class_names)
        
        # 保存JSON文件
        json_file = labelme_path / f"{txt_file.stem}.json"
        with open(json_file, "w") as f:
            json.dump(labelme_data, f, indent=2)
        
        # 复制图像文件
        shutil.copy2(image_file, labelme_path / image_file.name)

def visualize_yolo_obb_annotations(image_path, annotation_path, class_names, output_path):
    """
    可视化YOLO OBB标注
    
    Args:
        image_path: 原始图像路径
        annotation_path: YOLO格式的标注文件路径
        class_names: 类别名称列表
        output_path: 可视化结果保存路径
    """
    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"警告：无法读取图像 {image_path}")
        return
        
    height, width = image.shape[:2]
    
    # 为每个类别分配不同的颜色
    colors = [
        (0, 255, 0),    # 绿色
        (255, 0, 0),    # 蓝色
        (0, 0, 255),    # 红色
        (255, 255, 0),  # 青色
        (255, 0, 255),  # 粉色
        (0, 255, 255),  # 黄色
    ]
    
    # 读取标注
    with open(annotation_path, "r") as f:
        annotations = f.readlines()
    
    # 绘制每个标注
    for ann in annotations:
        parts = ann.strip().split()
        class_id = int(parts[0])
        
        # 获取坐标值
        coords = []
        for v in parts[1:9]:  # 只取8个坐标
            v = v.strip('[],')
            if v:
                coords.append(float(v))
        points = np.array(coords, dtype=np.float32).reshape(-1, 2)
        points = points.astype(np.int32)
        
        # 绘制旋转框
        color = colors[class_id % len(colors)]
        cv2.drawContours(image, [points], 0, color, 2)
        
        # 添加类别标签
        label = class_names[class_id]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        
        # 计算文本背景框的位置
        text_x = points[0][0]
        text_y = points[0][1] - 5
        
        # 绘制文本背景
        cv2.rectangle(image, 
                     (text_x, text_y - text_size[1] - 5),
                     (text_x + text_size[0], text_y),
                     color, -1)
        
        # 绘制文本
        cv2.putText(image, label,
                    (text_x, text_y),
                    font, font_scale, (255, 255, 255), thickness)
    
    # 保存结果
    cv2.imwrite(str(output_path), image)

def main():
    # yolo_obb_path = r"D:\downloads\create_obb_label\yolo\C4DA10"
    yolo_obb_path = r"C:\Users\Eugene\Desktop\golden_wire_dataset\baikang_ball_lead_pad_dataset\lead_seg\image"
    
    labelme_path = r"C:\Users\Eugene\Desktop\golden_wire_dataset\baikang_ball_lead_pad_dataset\lead_seg\image"
    class_names = ["lead"]  # 请确保这个列表与您的类别一致

    process_dataset(yolo_obb_path, labelme_path, class_names)
    # print("转换完成！LabelMe格式的JSON文件和图像已保存到指定目录。")

    # 可视化YOLO OBB标注
    # output_dir = Path(yolo_obb_path).parent / "train_data_vis_yolo"
    # output_dir.mkdir(parents=True, exist_ok=True)
    # txt_files = list(Path(yolo_obb_path).glob("*.txt"))
    # for txt_file in txt_files:
    #     image_file = txt_file.with_suffix(".jpg")
    #     image_file = Path(str(image_file).replace("org_imgs_labels_v6.6", "org_imgs"))
    #     if not image_file.exists():
    #         image_file = txt_file.with_suffix(".png")
    #         image_file = Path(str(image_file).replace("org_imgs_labels_v6.6", "org_imgs"))
        
    #     if not image_file.exists():
    #         print(f"Warning: Image file not found for {image_file}")
    #         continue
        
    #     output_path = output_dir / f"{txt_file.stem}_visualized.jpg"
    #     visualize_yolo_obb_annotations(image_file, txt_file, class_names, output_path)

if __name__ == "__main__":
    main()
