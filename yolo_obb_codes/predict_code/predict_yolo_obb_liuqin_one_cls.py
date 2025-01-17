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

def process_image(image_path, output_path, visual_path, gt_label_path, model, imgsz, target_cls=2):
    """处理单张图像的预测和可视化
    Args:
        image_path: 输入图像路径
        output_path: 输出标签路径
        visual_path: 可视化结果保存路径
        gt_label_path: GT标签路径
        model: YOLO模型
        imgsz: 图像大小
        target_cls: 目标类别ID，默认为2
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error reading image: {image_path}")
        return
    
    img_height, img_width = img.shape[:2]
    
    # 进行预测
    results = model(img, imgsz=imgsz, verbose=False)[0]
    
    # 过滤预测框，只保留目标类别的预测框
    pred_boxes = []
    for obb in results.obb:
        cls_id = int(obb.cls)
        if cls_id == target_cls:  # 只保留目标类别的预测框
            coords = obb.xyxyxyxy.tolist()[0]
            pred_boxes.append({
                'coords': coords,
                'cls': cls_id,
                'conf': float(obb.conf),
                'matched': False
            })
    
    # 获取GT框列表并过滤
    gt_boxes = read_gt_label(gt_label_path, img_width, img_height)
    gt_boxes = [box for box in gt_boxes if box['cls'] == target_cls]  # 只保留目标类别的GT框
    
    # 计算IoU并标记匹配的框
    iou_threshold = 0.3
    
    # 匹配GT框和预测框
    for gt_idx, gt_box in enumerate(gt_boxes):
        max_iou = 0
        best_pred_idx = -1
        
        # 找到与当前GT框IoU最大的预测框
        for pred_idx, pred_box in enumerate(pred_boxes):
            if not pred_box['matched']:
                iou = calculate_iou(gt_box['coords'], pred_box['coords'])
                if iou > max_iou:
                    max_iou = iou
                    best_pred_idx = pred_idx
        
        # 处理匹配结果
        if max_iou >= iou_threshold:
            gt_boxes[gt_idx]['matched'] = True
            pred_boxes[best_pred_idx]['matched'] = True
    
    # 绘制目标框并保存结果
    img_result = img.copy()
    
    # 绘制预测框
    for pred_box in pred_boxes:
        if pred_box['matched']:
            # 正确检测 - 绿色
            color = (0, 255, 0)
        else:
            # 误检 - 紫色
            color = (255, 0, 255)
        draw_obb(img_result, pred_box['coords'], color=color, thickness=2)
        
        # 修改这里：使用第一个点的坐标
        # text_x = int(pred_box['coords'][0])  # 第一个点的x坐标
        # text_y = int(pred_box['coords'][1])  # 第一个点的y坐标
        # text_pos = (text_x, text_y - 10)
        
        # cv2.putText(img_result, f"conf:{pred_box['conf']:.2f}", text_pos, 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 绘制未匹配的GT框（漏检）
    for gt_box in gt_boxes:
        if not gt_box['matched']:
            # 漏检 - 黄色
            color = (0, 255, 255)
            draw_obb(img_result, gt_box['coords'], color=color, thickness=2)
            
    # 将原图拼接在结果图右边
    h, w = img_result.shape[:2]
    combined_img = np.zeros((h, w*2, 3), dtype=np.uint8)
    combined_img[:, :w] = img_result
    combined_img[:, w:] = img
    img_result = combined_img
    # 保存可视化结果
    cv2.imwrite(visual_path, img_result)

    # 如果需要统计属性，可以返回相关信息
    return {
        'total_gt': len(gt_boxes),
        'total_pred': len(pred_boxes),
        'matched': sum(1 for box in pred_boxes if box['matched']),
        'false_positives': sum(1 for box in pred_boxes if not box['matched']),
        'false_negatives': sum(1 for box in gt_boxes if not box['matched'])
    }

def predict_images(input_dir, output_dir, model_path, gt_label_dir=None, imgsz=2560, target_cls=2):
    """
    批量预测图像并保存结果
    
    Args:
        input_dir (str): 输入图像目录
        output_dir (str): 输出结果目录
        model_path (str): 模型路径
        gt_label_dir (str, optional): 真实标签目录
        imgsz (int, optional): 图像大小，默认2560
        target_cls (int, optional): 目标类别ID，默认2
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

    #image_files = image_files[0:40]
    
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
            image_path, 
            output_label_path, 
            output_visual_path, 
            gt_label_path, 
            model, 
            imgsz,
            target_cls=target_cls  # 传入目标类别
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
    #model_path = '/mnt/nfs/AOI_detection/ccm/models/v6/yolo11s_obb_2560/liuqin_debug/yolo11s_obb_with_syn/20250110_152451/exp_s_2560_20250110_152451/train/weights/best.pt'
    #model_path =  '/mnt/nfs/AOI_detection/ccm/models/v6/yolo11s_obb_2560/liuqin_debug/yolo11s_obb_with_syn_only_train/20250110_165019/exp_s_2560_20250110_165019/train/weights/best.pt'
    model_path =  '/mnt/nfs/AOI_detection/ccm/models/v6/yolo11s_obb_2560/liuqin_debug/yolo11s_obb_with_syn_trans_cls2/20250115_153557/exp_s_2560_20250115_153557/train/weights/best.pt'
    #model_path = '/mnt/nfs/AOI_detection/ccm/models/v6/yolo11s_obb_2560/liuqin_debug/yolo11s_obb_without_syn_trans_cls2/20250115_153824/exp_s_2560_20250115_153824/train/weights/best.pt'
    name = 'with_syn_one_cls_trans_cls2_glue_combine'
    input_dir = f'/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/liuqin_data/test_dataset_transformed_cls2/images/train'
    gt_label_dir = f'/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/liuqin_data/test_dataset_transformed_cls2/labels/train'  # 可选，如不需要设为None
    output_dir = f'/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/liuqin_data/test_dataset_transformed_cls2/predicted_result/{name}/train/'
    imgsz = 2560
    target_cls = 1 # 设置目标类别
    
    # 执行预测
    predict_images(
        input_dir=input_dir,
        output_dir=output_dir,
        model_path=model_path,
        gt_label_dir=gt_label_dir,
        imgsz=imgsz,
        target_cls=target_cls  # 传入目标类别
    )


    
if __name__ == "__main__":
    main()
