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

def merge_close_dirt_shapes(shapes, distance_threshold=20, min_area=100):
    """
    合并距离较近的dirt类标注
    
    Args:
        shapes: 标注列表
        distance_threshold: 距离阈值，小于此距离的标注将被合并
        min_area: 最小面积阈值，小于此面积的标注将被忽略
    Returns:
        merged_shapes: 合并后的标注列表
    """
    dirt_shapes = [s for s in shapes if s["label"] == "dirt"]
    other_shapes = [s for s in shapes if s["label"] != "dirt"]
    
    if len(dirt_shapes) <= 1:
        return shapes
    
    # 计算所有dirt标注的中心点
    centers = []
    for shape in dirt_shapes:
        points = np.array(shape["points"])
        center = points.mean(axis=0)
        centers.append(center)
    centers = np.array(centers)
    
    # 构建合并组
    merged_groups = []
    used = set()
    
    for i in range(len(dirt_shapes)):
        if i in used:
            continue
            
        current_group = [i]
        used.add(i)
        
        # 查找与当前标注距离较近的其他标注
        for j in range(i + 1, len(dirt_shapes)):
            if j in used:
                continue
            
            # 计算两个标注中心点之间的距离
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < distance_threshold:
                current_group.append(j)
                used.add(j)
        
        merged_groups.append(current_group)
    
    # 合并每个组内的标注
    merged_dirt_shapes = []
    for group in merged_groups:
        if len(group) == 1:
            # 对单个标注也检查面积
            points = np.array(dirt_shapes[group[0]]["points"])
            area = cv2.contourArea(points)
            if area >= min_area:  # 设置最小面积阈值
                merged_dirt_shapes.append(dirt_shapes[group[0]])
        else:
            # 获取组内所有标注的点
            all_points = []
            for idx in group:
                points = np.array(dirt_shapes[idx]["points"])
                all_points.extend(points)
            all_points = np.array(all_points)
            
            # 计算包围所有点的最小矩形
            rect = cv2.minAreaRect(all_points)
            box = cv2.boxPoints(rect)
            
            # 计算合并后的面积
            area = cv2.contourArea(box)
            
            # 只有面积大于阈值时才保留
            if area >= min_area:  # 设置最小面积阈值
                # 创建新的合并标注
                merged_shape = dirt_shapes[group[0]].copy()
                merged_shape["points"] = box.tolist()
                merged_shape["shape_type"] = "polygon"  # 合并后统一使用多边形类型
                
                merged_dirt_shapes.append(merged_shape)
                # print(f"合并了 {len(group)} 个dirt标注，面积: {area:.2f}")
            # else:
            #     print(f"忽略面积过小的合并标注，面积: {area:.2f}")
    
    return other_shapes + merged_dirt_shapes

def filter_annotations_by_difference(image_path, shapes, kernel_size=5, contrast_threshold=20, merge_distance=40, min_area=100):
    """
    基于局部对比度过滤标注区域，并分离稀疏的dirt标注，然后合并相近的dirt标注
    
    Args:
        image_path: 图像文件路径
        shapes: labelme格式的标注列表
        kernel_size: 中值滤波核大小，默认5
        contrast_threshold: 对比度阈值，默认20
        merge_distance: dirt标注合并距离阈值，默认20
    
    Returns:
        filtered_shapes: 过滤和合并后的标注列表
    """
    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"警告：无法读取图像 {image_path}")
        return shapes
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 中值滤波
    blurred = cv2.medianBlur(gray, kernel_size)
    
    # 计算差值图
    diff = cv2.absdiff(gray, blurred)
    
    filtered_shapes = []
    for shape in shapes:
        points = np.array(shape["points"], dtype=np.int32)
        
        # 创建标注区域的mask
        region_mask = np.zeros_like(gray, dtype=np.uint8)
        if shape["shape_type"] == "circle" and len(points) == 2:
            center = ((points[0][0] + points[1][0]) // 2, (points[0][1] + points[1][1]) // 2)
            radius = int(max(abs(points[1][0] - points[0][0]), abs(points[1][1] - points[0][1])) // 2)
            cv2.circle(region_mask, center, radius, 255, -1)
        else:
            cv2.fillPoly(region_mask, [points], 255)
        
        # 获取该区域内的差值
        region_diff = diff.copy()
        region_diff[region_mask == 0] = 0  # 将区域外的差值置为0
        
        # # 保存调试图像
        # debug_mask = np.zeros_like(gray)
        # debug_mask[region_mask > 0] = diff[region_mask > 0]
        # cv2.imwrite(f"debug_diff_{shapes.index(shape)}.png", debug_mask)

        # 对dirt类型进行连通域分析
        if shape["label"] == "dirt":
            # 创建高对比度区域的mask
            high_contrast_mask = (region_diff > contrast_threshold).astype(np.uint8)
            
            # 进行连通域分析
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                high_contrast_mask, connectivity=8
            )
            
            # 处理每个连通域
            valid_regions = []
            for i in range(1, num_labels):  # 跳过背景（标签0）
                area = stats[i, cv2.CC_STAT_AREA]
                if area < min_area:  # 过滤掉太小的连通域
                    continue
                    
                # 获取当前连通域的边界框
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                # 创建新的shape
                new_shape = shape.copy()
                if shape["shape_type"] == "circle":
                    # 对于圆形，使用边界框的中心和半径
                    center_x = x + w/2
                    center_y = y + h/2
                    radius = max(w, h)/2
                    new_shape["points"] = [
                        [center_x - radius, center_y - radius],
                        [center_x + radius, center_y + radius]
                    ]
                else:
                    # 对于多边形，使用边界框的四个角点
                    new_shape["points"] = [
                        [x, y],
                        [x + w, y],
                        [x + w, y + h],
                        [x, y + h]
                    ]
                
                # 计算该区域的最大对比度
                region_mask_i = (labels == i).astype(np.uint8)
                max_contrast = np.max(region_diff[region_mask_i > 0])
                
                # print(f"dirt标注 {shapes.index(shape)} 的子区域 {i}, 对比度: {max_contrast}")
                
                if max_contrast > contrast_threshold:
                    valid_regions.append(new_shape)
            
            # 如果找到有效的子区域，添加到结果中
            if valid_regions:
                filtered_shapes.extend(valid_regions)
                # print(f"dirt标注 {shapes.index(shape)} 分离为 {len(valid_regions)} 个子区域")
            else:
                # print(f"dirt标注 {shapes.index(shape)} 被过滤掉")
                pass
        
        # 对其他类型只进行对比度分析
        else:
            max_contrast = np.max(region_diff)
            if max_contrast > contrast_threshold:
                filtered_shapes.append(shape)
                # print(f"保留{shape['label']}标注 {shapes.index(shape)}, 对比度: {max_contrast}")
            else:
                # print(f"过滤{shape['label']}标注 {shapes.index(shape)}, 对比度: {max_contrast}")
                pass
    
    # 在返回结果之前，合并相近的dirt标注
    filtered_shapes = merge_close_dirt_shapes(filtered_shapes, merge_distance, min_area)
    
    return filtered_shapes

def convert_labelme_to_yolo_obb(json_file, class_names, threshold=20, kernel_size=5, merge_distance=40, min_area=100):
    with open(json_file, "r") as f:
        data = json.load(f)
    
    # 获取对应的图像文件路径
    image_path = Path(json_file).with_suffix('.jpg')
    
    # 应用过滤器
    # filtered_shapes = filter_annotations_by_difference(image_path, data["shapes"], kernel_size, threshold, merge_distance, min_area)
    filtered_shapes = data["shapes"]
    image_width = data["imageWidth"]
    image_height = data["imageHeight"]
    
    yolo_annotations = []
    for shape in filtered_shapes:
        label = shape["label"]
        if label not in class_names:
            continue
        class_id = class_names.index(label)
        points = np.array(shape["points"], dtype=np.float32)
        
        if shape["shape_type"] == "circle":
            # 处理圆形目标
            if len(points) == 2:
                # 假设两个点分别是左上角和右下角
                x1, y1 = points[0]
                x2, y2 = points[1]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                radius = max(abs(x2 - x1), abs(y2 - y1)) / 2
                
                # 生成圆的四个角点
                corners = np.array([
                    [center_x - radius, center_y - radius],
                    [center_x + radius, center_y - radius],
                    [center_x + radius, center_y + radius],
                    [center_x - radius, center_y + radius]
                ])
            else:
                print(f"警告：在文件 {json_file} 中，圆形标签 {label} 的点数不正确，跳过此标注。")
                continue
        else:
            # 处理其他形状（多边形或矩形）
            if len(points) < 4:
                print(f"警告：在文件 {json_file} 中，标签 {label} 的点数少于4个，跳过此标注。")
                continue
            
            # 计算最小面积矩形
            rect = cv2.minAreaRect(points)
            corners = cv2.boxPoints(rect)
        
        # 归一化坐标
        normalized_corners = corners / [image_width, image_height]
        
        # 格式化为 YOLO OBB 格式
        obb_annotation = f"{class_id} " + " ".join([f"{x:.6f} {y:.6f}" for x, y in normalized_corners])
        yolo_annotations.append(obb_annotation)
    
    return yolo_annotations

def process_dataset(labelme_path, yolo_path, class_names, test_mode=False, threshold=20, kernel_size=5, merge_distance=40, min_area=100):
    images_path = Path(yolo_path) / "images"
    labels_path = Path(yolo_path) / "labels"
    vis_path = Path(yolo_path) / "visualizations"
    
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)
    os.makedirs(vis_path, exist_ok=True)
    
    json_files = list(Path(labelme_path).glob("*.json"))
    # json_files = list(Path(labelme_path).glob("582370_顶部2通讯_组合采图.json"))    
    # json_files = list(Path(labelme_path).glob("582383_顶部2通讯_组合采图10.json"))    
    if test_mode:
        json_files = random.sample(json_files, min(5, len(json_files)))
    
    for json_file in tqdm(json_files, desc="处理文件"):
        yolo_annotations = convert_labelme_to_yolo_obb(json_file, class_names, threshold, kernel_size, merge_distance, min_area)
        
        label_file = labels_path / f"{json_file.stem}.txt"
        with open(label_file, "w") as f:
            f.write("\n".join(yolo_annotations))
        
        image_file = json_file.with_suffix(".jpg")
        if image_file.exists():
            shutil.copy(image_file, images_path / image_file.name)
            
            vis_output = vis_path / f"vis_{image_file.name}"
            visualize_yolo_obb_annotations(
                image_file,
                label_file,
                class_names,
                vis_output
            )

def split_dataset(yolo_path, train_ratio=0.8):
    images_path = Path(yolo_path) / "images"
    labels_path = Path(yolo_path) / "labels"
    
    all_files = list(labels_path.glob("*.txt"))
    random.seed(42)
    random.shuffle(all_files)
    
    split_index = int(len(all_files) * train_ratio)
    train_files = all_files[:split_index]
    val_files = all_files[split_index:]
    
    for split, files in [("train", train_files), ("val", val_files)]:
        (images_path / split).mkdir(exist_ok=True)
        (labels_path / split).mkdir(exist_ok=True)
        
        for file in tqdm(files, desc=f"移动{split}文件"):
            shutil.move(file, labels_path / split / file.name)
            image_file = images_path / file.with_suffix(".jpg").name
            if image_file.exists():
                shutil.move(image_file, images_path / split / image_file.name)

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
        points = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
        
        # 反归一化坐标
        points[:, 0] *= width
        points[:, 1] *= height
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

def test_conversion(labelme_path, test_output_path, class_names, threshold=20, kernel_size=5, merge_distance=40, min_area=100):
    yolo_obb_test_path = Path(test_output_path) / "yolo_obb_data"
    visualization_path = Path(test_output_path) / "obb_visualizations"
    
    os.makedirs(yolo_obb_test_path, exist_ok=True)
    os.makedirs(visualization_path, exist_ok=True)
    
    process_dataset(labelme_path, yolo_obb_test_path, class_names, test_mode=True, 
                   threshold=threshold, kernel_size=kernel_size, merge_distance=merge_distance, min_area=min_area)
    
    for annotation_file in tqdm(list(Path(yolo_obb_test_path / "labels").glob("*.txt")), desc="生成可视化"):
        image_file = yolo_obb_test_path / "images" / annotation_file.with_suffix(".jpg").name
        output_image = visualization_path / f"visualized_obb_{image_file.name}"
        visualize_yolo_obb_annotations(image_file, annotation_file, class_names, output_image)
    
    print(f"YOLO格式的旋转目标框检测测试数据已保存到 {yolo_obb_test_path}")
    print(f"旋转目标框检测可视化结果已保存到 {visualization_path}")

def generate_dataset_yaml(yolo_obb_path, class_names):
    yaml_content = {
        'path': str(yolo_obb_path),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    yaml_path = Path(yolo_obb_path) / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Dataset YAML 文件已生成：{yaml_path}")

def main(test_mode=False):
    labelme_path = "/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/test_data/after_check"
    yolo_obb_path = "/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/test_data/after_check_yolo_new"
    test_output_path = "/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/test_data/"
    class_names = ["dust", "glue", "dirt"]
    
    threshold = 20  # 对比度阈值
    kernel_size = 81  # 中值滤波核大小
    merge_distance = 100  # dirt标注合并距离阈值
    min_area = 100  # 最小面积阈值
    
    if test_mode:
        test_conversion(labelme_path, test_output_path, class_names, threshold, kernel_size, merge_distance, min_area)
    else:
        process_dataset(labelme_path, yolo_obb_path, class_names, test_mode=False, threshold=threshold, kernel_size=kernel_size, merge_distance=merge_distance, min_area=min_area)
        #split_dataset(yolo_obb_path)
        #generate_dataset_yaml(yolo_obb_path, class_names)
        print("旋转目标框检测数据转换、拆分和YAML生成完成！")

if __name__ == "__main__":
    main(test_mode=False)  # 设置为 False 进行正式转换