import os
import shutil
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import os.path as osp
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import pandas as pd

def draw_obb(img, obb, color=(0, 255, 0), thickness=2):
    pts = np.array(obb, np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, color, thickness)

def calculate_iou(box1, box2):
    """计算两个目标框的IoU"""
    try:
        # box1格式: [x1,y1,x2,y2,x3,y3,x4,y4]
        # box2格式: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        
        # 转换box1为多边形
        poly1_points = [(box1[i], box1[i+1]) for i in range(0, len(box1), 2)]
        poly1 = Polygon(poly1_points)
        
        # 转换box2为多边形
        poly2_points = [(point[0], point[1]) for point in box2]
        poly2 = Polygon(poly2_points)
        
        if not poly1.is_valid or not poly2.is_valid:
            return 0
        
        # 计算交并比
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        
        return intersection / union if union > 0 else 0
    except Exception as e:
        print(f"Error calculating IoU: {e}")
        print(f"Box1: {box1}")
        print(f"Box2: {box2}")
        return 0

def read_gt_label(label_path, img_width, img_height):
    """读取真实标签文件并转换为绝对坐标"""
    boxes = []
    if label_path is None or not osp.exists(label_path):
        return boxes
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                values = line.strip().split()
                if len(values) >= 9:  # class x1 y1 x2 y2 x3 y3 x4 y4
                    cls = int(float(values[0]))
                    coords = list(map(float, values[1:9]))
                    # 转换相对坐标到绝对坐标
                    abs_coords = []
                    for i in range(0, len(coords), 2):
                        abs_coords.append(coords[i] * img_width)
                        abs_coords.append(coords[i + 1] * img_height)
                    boxes.append({
                        'coords': abs_coords,
                        'cls': cls,
                        'matched': False
                    })
    except Exception as e:
        print(f"Error reading GT label: {e}")
    return boxes

def process_image(image_path, output_path, visual_path, gt_label_path, model, imgsz):
    """处理单张图像的预测和可视化"""
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error reading image: {image_path}")
        return
    
    img_height, img_width = img.shape[:2]
    
    # 进行预测
    #results = model(img, imgsz=imgsz, verbose=False, conf=0.1)[0]
    results = model(img, imgsz=imgsz, verbose=False)[0]
    # 获取预测框列表
    pred_boxes = []
    for obb in results.obb:
        coords = obb.xyxyxyxy.tolist()[0]
        pred_boxes.append({
            'coords': coords,
            'cls': int(obb.cls),
            'conf': float(obb.conf),
            'matched': False
        })
    
    # 获取GT框列表
    gt_boxes = read_gt_label(gt_label_path, img_width, img_height)
    
    # 计算IoU并标记匹配的框
    iou_threshold = 0.3
    current_name = None
    
    for gt_idx, gt_box in enumerate(gt_boxes):
        max_iou = 0
        best_pred_idx = -1
        best_pred_different_cls_idx = -1
        max_iou_different_cls = 0
        
        # 找到与当前GT框IoU最大的预测框（分别记录同类和不同类的情况）
        for pred_idx, pred_box in enumerate(pred_boxes):
            if not pred_box['matched']:
                iou = calculate_iou(gt_box['coords'], pred_box['coords'])
                
                if gt_box['cls'] == pred_box['cls']:
                    # 同类别情况
                    if iou > max_iou:
                        max_iou = iou
                        best_pred_idx = pred_idx
                else:
                    # 不同类别情况
                    if iou > max_iou_different_cls:
                        max_iou_different_cls = iou
                        best_pred_different_cls_idx = pred_idx
        
        # 优先处理同类别匹配
        if max_iou >= iou_threshold:
            gt_boxes[gt_idx]['matched'] = True
            pred_boxes[best_pred_idx]['matched'] = True
        # 如果没有同类别匹配，但有不同类别的高IoU匹配
        elif max_iou_different_cls >= iou_threshold:
            gt_boxes[gt_idx]['matched'] = True
            gt_boxes[gt_idx]['matched_different_cls'] = True
            pred_boxes[best_pred_different_cls_idx]['matched'] = True
            pred_boxes[best_pred_different_cls_idx]['matched_different_cls'] = True
            # 记录匹配信息用于调试
            print(f"Class mismatch: GT class={gt_box['cls']}, "
                  f"Pred class={pred_boxes[best_pred_different_cls_idx]['cls']}, "
                  f"IoU={max_iou_different_cls:.4f}")
    
    # 定义预测框的类别颜色映射 (BGR格式)
    # pred_class_colors = {
    #     0: (255, 128, 0),    # 深蓝色
    #     1: (128, 0, 255),    # 紫色
    #     2: (0, 128, 255),    # 橙色
    #     3: (255, 0, 128),    # 粉色
    #     4: (128, 255, 0),    # 青色
    #     5: (0, 255, 128),    # 浅绿色
    # }
    pred_class_colors = {
        0: (0, 255, 0),    # 绿色
        1: (0, 255, 0),    # 绿色
        2: (0, 255, 0)     # 绿色
    }
    
    # 绘制目标框并保存结果
    img_result = img.copy()
    
    # 先绘制预测框
    for pred_box in pred_boxes:
        if pred_box.get('matched_different_cls', False): # 尝试从 pred_box 字典中获取 key 为 'matched_different_cls' 的值，如果这个 key 不存在，则返回默认值 False
            # 类别不匹配的预测框 - 蓝色
            color = (255, 0, 0)  # BGR格式的蓝色
            # 使用预测框的第一个点坐标作为文本位置
            text_pos = (int(pred_box['coords'][0][0]), int(pred_box['coords'][0][1]) - 10)
            cv2.putText(img_result, f"Pred_cls:{pred_box['cls']}", text_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        elif pred_box['matched'] or len(gt_boxes) == 0:
            # 正确检测 - 使用类别对应的颜色
            color = pred_class_colors.get(pred_box['cls'], (0, 255, 0))
        else:
            # 误检 - 紫色
            color = (255, 0, 255)
        draw_obb(img_result, pred_box['coords'], color=color, thickness=2)
    
    # 再绘制未匹配的GT框（漏检）或类别不匹配的GT框
    properties_list = []
    is_missed = False
    is_different_cls = False
    ok = False
    for gt_box in gt_boxes:
        if not gt_box['matched']:
            # 漏检 - 黄色
            color = (0, 255, 255)

            if gt_box['cls'] == 0:
                is_missed = True
        
        elif gt_box.get('matched_different_cls', False):
            # 类别不匹配的GT框 - 橙色
            color = (0, 165, 255)  # BGR格式的橙色
            # 在框附近添加真实类别标注
            text_pos = (int(gt_box['coords'][0]), int(gt_box['coords'][1]) - 30)
            cv2.putText(img_result, f"GT_cls:{gt_box['cls']}", text_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            is_different_cls = True
        else:
            # 正确检测的GT框 - 灰色
            color = (128, 128, 128)
            ok = True

        if not gt_box['matched'] or gt_box.get('matched_different_cls', False):
            draw_obb(img_result, gt_box['coords'], color=color, thickness=2)
        else:
            draw_obb(img_result, gt_box['coords'], color=color, thickness=1)

        if gt_box['cls'] == 0:
            # 统计当前obb的面积、长宽比、亮度中值、亮度相对周边的对比度
            # 计算面积
            # 将一维坐标数组重塑为N×2的点坐标数组
            coords = np.array(gt_box['coords']).astype(np.float32).reshape(-1, 2)
            area = cv2.contourArea(coords)
            
            # 计算长宽比
            rect = cv2.minAreaRect(coords)
            width = min(rect[1])
            height = max(rect[1]) 
            aspect_ratio = height / width if width > 0 else 0
            
            # 获取obb区域的mask
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [coords.astype(np.int32)], 1)
            
            # 计算obb区域的亮度中值
            obb_pixels = img[mask == 1]
            if len(obb_pixels) > 0:
                brightness_median = np.median(obb_pixels)
            else:
                brightness_median = 0
                
            # 计算周边区域
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            dilated = cv2.dilate(mask, kernel, iterations=2)
            border = cv2.subtract(dilated, mask)
            
            # 计算周边区域的亮度中值
            border_pixels = img[border == 1]
            if len(border_pixels) > 0:
                border_brightness = np.median(border_pixels)
                # 计算对比度
                contrast = abs(float(brightness_median) - float(border_brightness))
            else:
                contrast = 0
                
            # 计算更多的统计特征
            # 1. 位置相关
            center_x = np.mean(coords[:, 0])
            center_y = np.mean(coords[:, 1])
            relative_x = center_x / img_width  # 归一化的x坐标
            relative_y = center_y / img_height # 归一化的y坐标
            
            # 2. 亮度相关
            obb_pixels = img[mask == 1]
            if len(obb_pixels) > 0:
                brightness_mean = np.mean(obb_pixels)
                brightness_std = np.std(obb_pixels)  # 内部亮度标准差
                brightness_min = np.min(obb_pixels)
                brightness_max = np.max(obb_pixels)
            else:
                brightness_mean = brightness_std = brightness_min = brightness_max = 0
            
            # 3. 形状相关
            perimeter = cv2.arcLength(coords, True)  # 周长
            compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0  # 圆形度
            
            # 4. 周边环境相关
            border_pixels = img[border == 1]
            if len(border_pixels) > 0:
                border_std = np.std(border_pixels)  # 周边区域亮度变化程度
                border_brightness_min = np.min(border_pixels)
                border_brightness_max = np.max(border_pixels)
            else:
                border_std = border_brightness_min = border_brightness_max = 0

            properties = {
                'filename': osp.basename(image_path),
                # 原有属性
                'area': area,
                'aspect_ratio': aspect_ratio,
                'brightness_median': brightness_median,
                'contrast': contrast,
                'is_missed': is_missed,
                'is_different_cls': is_different_cls,
                'ok': ok,
                # 新增位置属性
                'center_x': center_x,
                'center_y': center_y,
                'relative_x': relative_x,
                'relative_y': relative_y,
                # 新增亮度属性
                'brightness_mean': brightness_mean,
                'brightness_std': brightness_std,
                'brightness_min': brightness_min,
                'brightness_max': brightness_max,
                # 新增形状属性
                'perimeter': perimeter,
                'compactness': compactness,  # 越接近1表示越接近圆形
                # 新增周边环境属性
                'border_std': border_std,  # 周边区域亮度变化程度
                'border_brightness_min': border_brightness_min,
                'border_brightness_max': border_brightness_max,
                'brightness_range': brightness_max - brightness_min,  # dust内部亮度范围
                'border_range': border_brightness_max - border_brightness_min  # 周边亮度范围
            }
            properties_list.append(properties)
    
    # 写入预测结果
    # with open(output_path, 'w') as f:
    #     for pred_box in pred_boxes:
    #         line = f"{pred_box['cls']} {' '.join(map(str, pred_box['coords']))} {pred_box['conf']:.6f}\n"
    #         f.write(line)
    
        # 将原图拼接在结果图右边
    h, w = img_result.shape[:2]
    combined_img = np.zeros((h, w*2, 3), dtype=np.uint8)
    combined_img[:, :w] = img_result
    combined_img[:, w:] = img
    img_result = combined_img
    # 保存可视化结果
    cv2.imwrite(visual_path, img_result)

    return properties_list

def predict_images(input_dir, output_dir, model_path, gt_label_dir=None, imgsz=2560):
    """
    批量预测图像并保存结果
    
    Args:
        input_dir (str): 输入图像目录
        output_dir (str): 输出结果目录
        model_path (str): 模型路径
        gt_label_dir (str, optional): 真实标签目录
        imgsz (int, optional): 图像大小，默认2560
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    model = YOLO(model_path)
    
    # 获取所有图像文件的列表
    image_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    
    # 初始化用于收集所有dust属性的列表
    all_properties = []

    # image_files = image_files[0:100]
    
    # 使用tqdm创建进度条
    for image_path in tqdm(image_files, desc="Processing images", unit="image"):
        # 生成新的文件名，避免重名
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_image_path = os.path.join(output_dir, f"{name}{ext}")
        output_label_path = os.path.join(output_dir, f"{name}.txt")
        output_visual_path = os.path.join(output_dir, f"{name}_visual.jpg")
        
        # 如果文件已存在，添加数字后缀
        counter = 1
        while os.path.exists(output_image_path) or os.path.exists(output_label_path) or os.path.exists(output_visual_path):
            output_image_path = os.path.join(output_dir, f"{name}_{counter}{ext}")
            output_label_path = os.path.join(output_dir, f"{name}_{counter}.txt")
            output_visual_path = os.path.join(output_dir, f"{name}_{counter}_visual.jpg")
            counter += 1
        
        # 获取对应的真实标签路径（如果存在）
        gt_label_path = None
        if gt_label_dir:
            name = osp.splitext(osp.basename(image_path))[0]
            gt_label_path = osp.join(gt_label_dir, f"{name}.txt")
        
        # 复制原始图像到输出目录
        # shutil.copy2(image_path, output_image_path)
        
        # 处理图像并保存标签结果和可视化结果
        properties_list = process_image(
            image_path, output_label_path, output_visual_path, gt_label_path, model, imgsz
        )
        
        # 收集所有dust的属性
        if properties_list:
            all_properties.extend(properties_list)
    
    # 保存所有dust属性到CSV文件
    # if all_properties:
    #     df = pd.DataFrame(all_properties)
    #     # 将csv文件的目录保存为output_dir的上层目录
    #     csv_dir = osp.dirname(output_dir)
    #     csv_path = osp.join(csv_dir, 'dust_statistics.csv')
    #     df.to_csv(csv_path, index=False)
    #     print(f"\nDust statistics have been saved to: {csv_path}")
    
    print("\nAll images have been processed.")

def main():
    # 配置参数
    #model_path =  '/mnt/nfs/AOI_detection/ccm/models/v6/yolo11s_obb_2560/liuqin_debug/yolo11s_obb_with_syn/20250110_152451/exp_s_2560_20250110_152451/train/weights/best.pt'
    #model_path =  '/mnt/nfs/AOI_detection/ccm/models/v6/yolo11s_obb_2560/liuqin_debug/yolo11s_obb_with_syn_only_train/20250110_165019/exp_s_2560_20250110_165019/train/weights/best.pt'
    #model_path =  '/mnt/nfs/AOI_detection/ccm/models/v6/yolo11s_obb_2560/liuqin_debug/yolo11s_obb_without_syn/20250110_152955/exp_s_2560_20250110_152955/train/weights/best.pt'
    #model_path =  '/mnt/nfs/AOI_detection/ccm/models/v6/yolo11s_obb_2560/liuqin_debug/yolo11s_obb_with_syn_trans_cls2/20250115_153557/exp_s_2560_20250115_153557/train/weights/best.pt'
    model_path = '/mnt/nfs/AOI_detection/ccm/models/v6/yolo11s_obb_2560/liuqin_debug/yolo11s_obb_without_syn_trans_cls2/20250115_153824/exp_s_2560_20250115_153824/train/weights/best.pt'
    name = 'without_syn_trans_cls2'
    input_dir = f'/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/liuqin_data/test_dataset_transformed_cls2/images/train'
    gt_label_dir = f'/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/liuqin_data/test_dataset_transformed_cls2/labels/train'  # 可选，如不需要设为None
    output_dir = f'/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/liuqin_data/test_dataset_transformed_cls2/predicted_result/{name}/train/'

    imgsz = 2560
    
    # 执行预测
    predict_images(
        input_dir=input_dir,
        output_dir=output_dir,
        model_path=model_path,
        gt_label_dir=gt_label_dir,
        imgsz=imgsz
    )    


    
if __name__ == "__main__":
    main()
