import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import cv2
from tqdm import tqdm
import os.path as osp
import pandas as pd

def draw_box(img, box, color=(0, 255, 0), thickness=2):
    """Draw an AABB box on the image"""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

def calculate_iou(box1, box2):
    """Calculate IoU between two AABB boxes"""
    # box format: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def read_gt_label(label_path, img_width, img_height):
    """Read ground truth label file and convert to absolute coordinates"""
    boxes = []
    if label_path is None or not osp.exists(label_path):
        return boxes
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                values = line.strip().split()
                if len(values) >= 5:  # class x_center y_center width height
                    cls = int(values[0])
                    x_center, y_center, width, height = map(float, values[1:5])
                    
                    # Convert from normalized coordinates to absolute coordinates
                    x_center *= img_width
                    y_center *= img_height
                    width *= img_width
                    height *= img_height
                    
                    # Convert from center format to corner format
                    x1 = x_center - width/2
                    y1 = y_center - height/2
                    x2 = x_center + width/2
                    y2 = y_center + height/2
                    
                    boxes.append({
                        'coords': [x1, y1, x2, y2],
                        'cls': cls,
                        'matched': False
                    })
    except Exception as e:
        print(f"Error reading GT label: {e}")
    return boxes

def process_image(image_path, output_path, visual_path, gt_label_path, model, imgsz):
    """Process a single image for prediction and visualization"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error reading image: {image_path}")
        return
    
    img_height, img_width = img.shape[:2]
    
    # Make prediction
    results = model(img, imgsz=imgsz, verbose=False)[0]
    
    # Get predicted boxes
    pred_boxes = []
    if results is not None and results.boxes is not None:
        for box in results.boxes:
            coords = box.xyxy.tolist()[0]  # Get box in xyxy format
            pred_boxes.append({
                'coords': coords,
                'cls': int(box.cls),
                'conf': float(box.conf),
                'matched': False
            })
    
    # Get ground truth boxes
    gt_boxes = read_gt_label(gt_label_path, img_width, img_height)
    
    # Calculate IoU and mark matching boxes
    iou_threshold = 0.3
    
    for gt_idx, gt_box in enumerate(gt_boxes):
        max_iou = 0
        best_pred_idx = -1
        best_pred_different_cls_idx = -1
        max_iou_different_cls = 0
        
        # Find prediction box with maximum IoU
        for pred_idx, pred_box in enumerate(pred_boxes):
            if not pred_box['matched']:
                iou = calculate_iou(gt_box['coords'], pred_box['coords'])
                
                if gt_box['cls'] == pred_box['cls']:
                    if iou > max_iou:
                        max_iou = iou
                        best_pred_idx = pred_idx
                else:
                    if iou > max_iou_different_cls:
                        max_iou_different_cls = iou
                        best_pred_different_cls_idx = pred_idx
        
        # Handle matching
        if max_iou >= iou_threshold:
            gt_boxes[gt_idx]['matched'] = True
            pred_boxes[best_pred_idx]['matched'] = True
        elif max_iou_different_cls >= iou_threshold:
            gt_boxes[gt_idx]['matched'] = True
            gt_boxes[gt_idx]['matched_different_cls'] = True
            pred_boxes[best_pred_different_cls_idx]['matched'] = True
            pred_boxes[best_pred_different_cls_idx]['matched_different_cls'] = True
    
    # Define colors for different classes (BGR format)
    pred_class_colors = {
        0: (255, 128, 0),    # Deep blue
        1: (128, 0, 255),    # Purple
    }
    
    # Draw boxes and save results
    img_result = img.copy()
    
    # Draw prediction boxes (only for class 0)
    for pred_box in pred_boxes:
        if pred_box['cls'] == 0:  # Only draw boxes for class 0 (dust)
            color = (255, 0, 0)  # Blue for predictions (BGR format)
            draw_box(img_result, pred_box['coords'], color=color, thickness=2)
    
    # Draw unmatched GT boxes
    properties_list = []
    is_missed = False
    is_different_cls = False
    ok = False
    
    for gt_box in gt_boxes:
        if not gt_box['matched']:
            color = (0, 0, 255)  # Red for missed detections (BGR format)
            if gt_box['cls'] == 0:
                is_missed = True
                draw_box(img_result, gt_box['coords'], color=color, thickness=2)
        else:
            ok = True

        if gt_box['cls'] == 0:
            # Calculate box properties
            x1, y1, x2, y2 = gt_box['coords']
            width = x2 - x1
            height = y2 - y1
            area = width * height
            aspect_ratio = height / width if width > 0 else 0
            
            # Create mask for the box region
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 1, -1)
            
            # Calculate brightness statistics
            box_pixels = img[mask == 1]
            if len(box_pixels) > 0:
                brightness_median = np.median(box_pixels)
                brightness_mean = np.mean(box_pixels)
                brightness_std = np.std(box_pixels)
                brightness_min = np.min(box_pixels)
                brightness_max = np.max(box_pixels)
            else:
                brightness_median = brightness_mean = brightness_std = brightness_min = brightness_max = 0
            
            # Calculate border region
            border_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            border_size = 5
            cv2.rectangle(border_mask, 
                         (int(x1-border_size), int(y1-border_size)), 
                         (int(x2+border_size), int(y2+border_size)), 1, -1)
            border_mask = cv2.subtract(border_mask, mask)
            
            # Calculate border statistics
            border_pixels = img[border_mask == 1]
            if len(border_pixels) > 0:
                border_brightness = np.median(border_pixels)
                border_std = np.std(border_pixels)
                border_brightness_min = np.min(border_pixels)
                border_brightness_max = np.max(border_pixels)
                contrast = abs(float(brightness_median) - float(border_brightness))
            else:
                border_brightness = border_std = border_brightness_min = border_brightness_max = contrast = 0

            # Collect properties
            properties = {
                'filename': osp.basename(image_path),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'brightness_median': brightness_median,
                'contrast': contrast,
                'is_missed': is_missed,
                'is_different_cls': is_different_cls,
                'ok': ok,
                'center_x': (x1 + x2) / 2,
                'center_y': (y1 + y2) / 2,
                'relative_x': ((x1 + x2) / 2) / img_width,
                'relative_y': ((y1 + y2) / 2) / img_height,
                'brightness_mean': brightness_mean,
                'brightness_std': brightness_std,
                'brightness_min': brightness_min,
                'brightness_max': brightness_max,
                'border_std': border_std,
                'border_brightness_min': border_brightness_min,
                'border_brightness_max': border_brightness_max,
                'brightness_range': brightness_max - brightness_min,
                'border_range': border_brightness_max - border_brightness_min
            }
            properties_list.append(properties)
    
    # Write prediction results
    with open(output_path, 'w') as f:
        for pred_box in pred_boxes:
            # Convert to YOLO format (class x_center y_center width height)
            x1, y1, x2, y2 = pred_box['coords']
            width = x2 - x1
            height = y2 - y1
            x_center = x1 + width/2
            y_center = y1 + height/2
            
            # Normalize coordinates
            x_center /= img_width
            y_center /= img_height
            width /= img_width
            height /= img_height
            
            line = f"{pred_box['cls']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {pred_box['conf']:.6f}\n"
            f.write(line)
    
    # Save visualization
    cv2.imwrite(visual_path, img_result)
    
    return properties_list

def predict_images(input_dir, output_dir, model_path, gt_label_dir=None, imgsz=2560):
    """
    Batch predict images and save results
    
    Args:
        input_dir (str): Input images directory
        output_dir (str): Output directory
        model_path (str): Model path
        gt_label_dir (str, optional): Ground truth labels directory
        imgsz (int, optional): Image size, default 2560
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = YOLO(model_path)
    
    # Get list of image files
    image_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    
    # Initialize list for collecting all properties
    all_properties = []
    image_files = image_files[0:100]
    # Process images with progress bar
    for image_path in tqdm(image_files, desc="Processing images", unit="image"):
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_image_path = os.path.join(output_dir, f"{name}{ext}")
        output_label_path = os.path.join(output_dir, f"{name}.txt")
        output_visual_path = os.path.join(output_dir, f"{name}_visual.jpg")
        
        # Handle file name conflicts
        counter = 1
        while os.path.exists(output_image_path) or os.path.exists(output_label_path) or os.path.exists(output_visual_path):
            output_image_path = os.path.join(output_dir, f"{name}_{counter}{ext}")
            output_label_path = os.path.join(output_dir, f"{name}_{counter}.txt")
            output_visual_path = os.path.join(output_dir, f"{name}_{counter}_visual.jpg")
            counter += 1
        
        # Get corresponding ground truth label path
        gt_label_path = None
        if gt_label_dir:
            name = osp.splitext(osp.basename(image_path))[0]
            gt_label_path = osp.join(gt_label_dir, f"{name}.txt")
        
        # Copy original image
        shutil.copy2(image_path, output_image_path)
        
        # Process image
        properties_list = process_image(
            image_path, output_label_path, output_visual_path, gt_label_path, model, imgsz
        )
        
        # Collect properties
        if properties_list:
            all_properties.extend(properties_list)
    
    # Save statistics to CSV
    if all_properties:
        df = pd.DataFrame(all_properties)
        csv_dir = osp.dirname(output_dir)
        csv_path = osp.join(csv_dir, 'dust_statistics.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nDust statistics have been saved to: {csv_path}")
    
    print("\nAll images have been processed.")

def main():
    # Configuration
    model_path = '/mnt/nfs/AOI_detection/ccm/models/v5/yolo11s_2560/v5.15/20241210_171103/exp_s_2560_20241210_171103/train/weights/best.pt'
    model_name = 'yolo11s_aabb_2560_filtered'
    input_dir = f'/mnt/nfs/AOI_detection/ccm/data/v5/after_check/yolo_obb_filtered/images/train'
    output_dir = f'/mnt/nfs/AOI_detection/ccm/data/v5/after_check/yolo_aabb_filtered_v5.15/predicted_labels/train/{model_name}'
    gt_label_dir = '/mnt/nfs/AOI_detection/ccm/data/v5/after_check/yolo_aabb_allinone_max_med_contrast15_area20_converted/labels/train'
    imgsz = 2560
    
    # Execute prediction
    predict_images(
        input_dir=input_dir,
        output_dir=output_dir,
        model_path=model_path,
        gt_label_dir=gt_label_dir,
        imgsz=imgsz
    )

if __name__ == "__main__":
    main()
