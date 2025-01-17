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
                    cls = int(values[0])
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
    
    # 对img 做3*3 的中值滤波
    img = cv2.medianBlur(img, 3)
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
    pred_class_colors = {
        0: (255, 128, 0),    # 深蓝色
        1: (128, 0, 255),    # 浅紫色
    }
    
    # 绘制目标框并保存结果
    # 创建一个宽度为原图2倍的画布
    img_result = np.zeros((img.shape[0], img.shape[1]*2, 3), dtype=np.uint8)
    # 将原图复制到左边
    img_result[:, :img.shape[1]] = img.copy()
    # 将原图复制到右边
    img_result[:, img.shape[1]:] = img.copy()
    
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
    
    # 统计漏检、类别不匹配、正确检测的GT框的对比度和面积等属性
    properties_list = []
    for gt_box in gt_boxes:
        is_missed = False
        is_different_cls = False
        ok = False
        if not gt_box['matched']:
            # 漏检 - 黄色
            color = (0, 255, 255)

            if gt_box['cls'] == 0:
                is_missed = True
        
        elif gt_box.get('matched_different_cls', False):
            # 类别不匹配的GT框 - 橙色
            color = (0, 165, 255) 
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
            # 统计当前obb的面积、对比度
            # 计算面积
            # 将一维坐标数组重塑为N×2的点坐标数组
            coords = np.array(gt_box['coords']).astype(np.float32).reshape(-1, 2)
            area = cv2.contourArea(coords)
            
            # 计算对比度
            # 获取obb区域的mask
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [coords.astype(np.int32)], 1)
            # 计算obb区域的亮度最大值
            obb_pixels = img[mask == 1]
            if len(obb_pixels) > 0:
                obb_brightness_max = np.max(obb_pixels)
            else:
                obb_brightness_max = 0
            # 计算周边区域
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            dilated = cv2.dilate(mask, kernel, iterations=2)
            border = cv2.subtract(dilated, mask)
            # 计算周边区域的亮度中值
            border_pixels = img[border == 1]
            if len(border_pixels) > 0:
                border_brightness_median = np.median(border_pixels)
                # 计算对比度
                contrast = abs(float(obb_brightness_max) - float(border_brightness_median))
            else:
                contrast = 0
            # 在图像上显示对比度值
            text_pos = (int(gt_box['coords'][0]), int(gt_box['coords'][1]) - 10)
            color = (0, 0, 255)
            cv2.putText(img_result, f"{contrast:.1f}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            properties = {
                'filename': osp.basename(image_path),
                'area': area,
                'contrast': contrast,
                'is_missed': is_missed, 
                'is_different_cls': is_different_cls,
                'ok': ok
            }
            properties_list.append(properties)
    
    # 写入预测结果
    # with open(output_path, 'w') as f:
    #     for pred_box in pred_boxes:
    #         line = f"{pred_box['cls']} {' '.join(map(str, pred_box['coords']))} {pred_box['conf']:.6f}\n"
    #         f.write(line)
    
    # 保存可视化结果
    cv2.imwrite(visual_path, img_result)

    return properties_list

# 分析漏检样本的统计特征
def analyze_missed_samples(all_properties, output_dir):
    """分析漏检样本的统计特征"""
    output_dir = osp.join(output_dir, 'missed_samples_analysis')
    os.makedirs(output_dir, exist_ok=True)
    # 将列表转换为DataFrame以便分析
    df = pd.DataFrame(all_properties)
    
    # 1. 生成漏检样本的对比度和面积直方图
    missed_samples = df[df['is_missed'] == True]
    
    # 对比度直方图
    contrast_step = 10
    plt.figure(figsize=(10, 6))
    plt.hist(missed_samples['contrast'], bins=range(0, 256, contrast_step), color='blue', alpha=0.7)
    plt.title('Contrast Distribution of Missed Samples', fontsize=12)
    plt.xlabel('Contrast', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.grid(True)
    contrast_hist_path = osp.join(output_dir, 'missed_samples_contrast_hist.png')
    plt.savefig(contrast_hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 面积直方图
    plt.figure(figsize=(10, 6))
    plt.hist(missed_samples['area'], bins=30, color='green', alpha=0.7)
    plt.title('Area Distribution of Missed Samples', fontsize=12)
    plt.xlabel('Area', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.grid(True)
    area_hist_path = osp.join(output_dir, 'missed_samples_area_hist.png')
    plt.savefig(area_hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 基于对比度阈值的分析
    contrast_threshold = 42  # 设置对比度阈值

    # 高对比度样本分析
    high_contrast_samples = df[df['contrast'] > contrast_threshold]
    high_contrast_missed = high_contrast_samples[high_contrast_samples['is_missed'] == True]
    high_contrast_ratio = len(high_contrast_missed) / len(high_contrast_samples) if len(high_contrast_samples) > 0 else 0
    
    # 低对比度样本分析
    low_contrast_samples = df[df['contrast'] <= contrast_threshold]
    low_contrast_missed = low_contrast_samples[low_contrast_samples['is_missed'] == True]
    low_contrast_ratio = len(low_contrast_missed) / len(low_contrast_samples) if len(low_contrast_samples) > 0 else 0
    
    # 打印分析结果
    print(f"\n对比度分析结果 (阈值={contrast_threshold}):")
    print(f"\n总样本数: {len(df)}")
    print(f"总漏检样本数: {len(missed_samples)}")
    print(f"总体漏检比例: {len(missed_samples)/len(df):.2%}")
    
    print(f"\n高对比度样本 (>{contrast_threshold}):")
    print(f"  - 总样本数: {len(high_contrast_samples)}")
    print(f"  - 漏检样本数: {len(high_contrast_missed)}")
    print(f"  - 漏检比例: {high_contrast_ratio:.2%}")
    
    print(f"\n低对比度样本 (≤{contrast_threshold}):")
    print(f"  - 总样本数: {len(low_contrast_samples)}")
    print(f"  - 漏检样本数: {len(low_contrast_missed)}")
    print(f"  - 漏检比例: {low_contrast_ratio:.2%}")
    
    # 3. 基于面积阈值的分析
    area_threshold = 25  # 设置面积阈值

    # 大面积样本分析
    large_area_samples = df[df['area'] > area_threshold]
    large_area_missed = large_area_samples[large_area_samples['is_missed'] == True]
    large_area_ratio = len(large_area_missed) / len(large_area_samples) if len(large_area_samples) > 0 else 0
    
    # 小面积样本分析
    small_area_samples = df[df['area'] <= area_threshold]
    small_area_missed = small_area_samples[small_area_samples['is_missed'] == True]
    small_area_ratio = len(small_area_missed) / len(small_area_samples) if len(small_area_samples) > 0 else 0
    
    # 打印面积分析结果
    print(f"\n面积分析结果 (阈值={area_threshold}):")
    print(f"\n总样本数: {len(df)}")
    print(f"总漏检样本数: {len(missed_samples)}")
    print(f"总体漏检比例: {len(missed_samples)/len(df):.2%}")
    
    print(f"\n大面积样本 (>{area_threshold}):")
    print(f"  - 总样本数: {len(large_area_samples)}")
    print(f"  - 漏检样本数: {len(large_area_missed)}")
    print(f"  - 漏检比例: {large_area_ratio:.2%}")
    
    print(f"\n小面积样本 (≤{area_threshold}):")
    print(f"  - 总样本数: {len(small_area_samples)}")
    print(f"  - 漏检样本数: {len(small_area_missed)}")
    print(f"  - 漏检比例: {small_area_ratio:.2%}")
    
    # 4. 基于对比度分段分析
    # 设置对比度区间，步长为10
    contrast_bins = list(range(0, 256, contrast_step))
    missed_ratios = []
    total_counts = []
    
    # 计算每个区间的漏检比例
    for i in range(len(contrast_bins)-1):
        start = contrast_bins[i]
        end = contrast_bins[i+1]
        
        # 获取该区间的样本
        interval_samples = df[(df['contrast'] >= start) & (df['contrast'] < end)]
        total_count = len(interval_samples)
        
        if total_count > 0:
            # 计算该区间内的漏检比例
            missed_count = len(interval_samples[interval_samples['is_missed'] == True])
            missed_ratio = missed_count / total_count
        else:
            missed_ratio = 0
            
        missed_ratios.append(missed_ratio)
        total_counts.append(total_count)
    
    # 绘制漏检比例直方图
    plt.figure(figsize=(15, 8))
    
    # 主要的漏检比例柱状图
    bars = plt.bar(contrast_bins[:-1], missed_ratios, width=10, alpha=0.6)
    
    # 在每个柱子上标注样本总数
    for i, (rect, count) in enumerate(zip(bars, total_counts)):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., height,
                f'n={count}\n{missed_ratios[i]:.1%}',
                ha='center', va='bottom', rotation=90)
    
    plt.title('Distribution of Miss Rate vs Contrast', fontsize=14)
    plt.xlabel('Contrast Range', fontsize=12)
    plt.ylabel('Miss Rate', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 设置x轴刻度
    plt.xticks(contrast_bins, rotation=45)
    
    # 保存图像
    contrast_analysis_path = osp.join(output_dir, 'contrast_missed_ratio_analysis.png')
    plt.savefig(contrast_analysis_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 基于面积分段分析
    area_step = 25  # 设置面积区间步长为50
    area_bins = list(range(0, 550, area_step))  # 0到800,步长50
    area_missed_ratios = []
    area_total_counts = []
    
    # 计算每个面积区间的漏检比例
    for i in range(len(area_bins)-1):
        start = area_bins[i]
        end = area_bins[i+1]
        
        # 获取该区间的样本
        interval_samples = df[(df['area'] >= start) & (df['area'] < end)]
        total_count = len(interval_samples)
        
        if total_count > 0:
            # 计算该区间内的漏检比例
            missed_count = len(interval_samples[interval_samples['is_missed'] == True])
            missed_ratio = missed_count / total_count
        else:
            missed_ratio = 0
            
        area_missed_ratios.append(missed_ratio)
        area_total_counts.append(total_count)
    
    # 绘制面积漏检比例直方图
    plt.figure(figsize=(15, 8))
    
    # 主要的漏检比例柱状图
    bars = plt.bar(area_bins[:-1], area_missed_ratios, width=area_step*0.8, alpha=0.6)
    
    # 在每个柱子上标注样本总数和漏检比例
    for i, (rect, count) in enumerate(zip(bars, area_total_counts)):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., height,
                f'n={count}\n{area_missed_ratios[i]:.1%}',
                ha='center', va='bottom', rotation=90)
    
    plt.title('Distribution of Miss Rate vs Area', fontsize=14)
    plt.xlabel('Area Range (pixels)', fontsize=12)
    plt.ylabel('Miss Rate', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 设置x轴刻度
    plt.xticks(area_bins, rotation=45)
    
    # 保存图像
    area_analysis_path = osp.join(output_dir, 'area_missed_ratio_analysis.png')
    plt.savefig(area_analysis_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    
def predict_images(input_dir, output_dir, model_path, gt_label_dir=None, imgsz=2560, is_test=False):
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

    if is_test:
        image_files = image_files[0:100]

    #################################  Predict  #################################
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
        #shutil.copy2(image_path, output_image_path)
        
        # 处理图像并保存标签结果和可视化结果
        properties_list = process_image(
            image_path, output_label_path, output_visual_path, gt_label_path, model, imgsz
        )
        
        # 收集所有dust的属性
        if properties_list:
            all_properties.extend(properties_list)
    
    #################################  Analysis  #################################
    # 保存所有dust属性到CSV文件
    if all_properties:
        df = pd.DataFrame(all_properties)
        # 将csv文件的目录保存为output_dir的上层目录
        csv_dir = osp.dirname(output_dir)
        # csv_path = osp.join(csv_dir, 'dust_statistics.csv')
        # df.to_csv(csv_path, index=False)
        # print(f"\nDust statistics have been saved to: {csv_path}")
        # 分析漏检样本的统计特征
        analyze_missed_samples(all_properties, csv_dir)
    
    print("\nAll images have been processed.")

def main():
    # 配置参数
    #model_folder = f'/mnt/nfs/AOI_detection/ccm/models/v5/liuqin/yolo11s_obb/yolo_obb_contrast20_area25_v1/20241219_153505/exp_s_2560_20241219_153505'
    #model_folder = f'/mnt/nfs/AOI_detection/ccm/models/v5/liuqin/yolo11s_obb/yolo_obb_contrast30_area30_v1/20241219_153755/exp_s_2560_20241219_153755'
    #model_folder = f'/mnt/nfs/AOI_detection/ccm/models/v5/liuqin/yolo11s_obb/yolo_obb_contrast42_area30_v1/20241219_154052/exp_s_2560_20241219_154052'
    #model_folder = f'/mnt/nfs/AOI_detection/ccm/models/v5/liuqin/yolo11s_obb/yolo_obb_contrast42_area30_augment5058_v1/20241220_101512/exp_s_2560_20241220_101512'
    #model_folder = f'/mnt/nfs/AOI_detection/ccm/models/v5/yolo11s_obb_2560/v6.10/20241218_155247/exp_s_2560_20241218_155247'
    model_folder = f'/mnt/nfs/AOI_detection/ccm/models/v5/liuqin/yolo11s_obb/yolo_obb_val_org/20241223_185739/exp_s_2560_20241223_185739'
    
    model_path = osp.join(model_folder, 'train/weights/best.pt')

    data_type = 'val'
    input_dir = f'/mnt/nfs/AOI_detection/ccm/data/v5/relabel_data/yolo_obb_forcheck_base/images/{data_type}'
    gt_label_dir = f'/mnt/nfs/AOI_detection/ccm/data/v5/relabel_data/yolo_obb_forcheck_base/labels/{data_type}'  # 可选，如不需要设为None
    
    output_dir = osp.join(model_folder, f'predict/{data_type}')
    imgsz = 2560
    
    # 执行预测
    predict_images(
        input_dir=input_dir,
        output_dir=output_dir,
        model_path=model_path,
        gt_label_dir=gt_label_dir,
        imgsz=imgsz,
        is_test=False
    )    

if __name__ == "__main__":
    main()
