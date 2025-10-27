import argparse
from ultralytics import YOLO
import yaml
import os
from datetime import datetime
import json

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def predict_yolo11(config):
    """使用YOLO11模型进行预测"""
    # 创建带时间戳的预测输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 获取输入文件夹名称
    input_folder_name = os.path.basename(os.path.normpath(config['source']))
    
    # 根据任务类型创建输出目录名称
    task_type = config.get('task', 'detect')
    predict_name = f"predict_{task_type}_{input_folder_name}_{config['img_size']}_{timestamp}"
    output_dir = os.path.join(config['output_base_dir'], timestamp, predict_name)
    os.makedirs(output_dir, exist_ok=True)

    # 保存使用的配置到输出目录
    config_output_path = os.path.join(output_dir, 'used_config.yaml')
    with open(config_output_path, 'w') as f:
        yaml.dump(config, f)

    # 初始化模型
    if config.get('model'):
        model = YOLO(config['model'])  # 使用指定的模型
        print(f"使用指定模型进行预测: {config['model']}")
    else:
        model = YOLO(config['model_path'])  # 使用默认模型
        print(f"使用默认模型进行预测: {config['model_path']}")

    # 设置预测参数
    # 注意：save为True时才保存图像，如果save_txt和show都为False，需要手动设置save=True
    save_predictions = config.get('save_txt', True) or config.get('save_conf', True) or True  # 总是保存预测图像
    
    predict_args = {
        "source": config['source'],  # 输入源（文件夹路径）
        "imgsz": config['img_size'],
        "batch": config['batch_size'],
        "device": config['device'],
        
        "project": output_dir,
        "name": "predictions",
        "exist_ok": True,
        "save": True,  # 强制保存预测图像
        
        # 预测特定参数
        "conf": config.get('conf_threshold', 0.25),  # 置信度阈值
        "iou": config.get('iou_threshold', 0.45),      # IoU阈值
        "max_det": config.get('max_det', 300),        # 每张图像最大检测数量（所有类别总和）
        
        # 保存选项
        "save_txt": config.get('save_txt', True),     # 保存预测结果为txt文件
        "save_conf": config.get('save_conf', True),   # 保存置信度
        "save_json": config.get('save_json', False),   # 保存为COCO格式JSON
        "save_crop": config.get('save_crop', False),  # 保存裁剪的预测框
        
        # 可视化选项
        "show": config.get('show', False),            # 显示结果（批量预测时建议设为false）
        "show_labels": config.get('show_labels', True),  # 显示标签
        "show_conf": config.get('show_conf', True),   # 显示置信度
        
        # 其他选项
        "verbose": config.get('verbose', True),       # 显示详细信息
        
        # ONNX特定设置（如果使用ONNX模型）
        "half": False,  # ONNX推理时不使用FP16
    }

    # 根据任务类型添加task参数
    task = config.get('task', 'detect')
    if task in ['detect', 'segment', 'pose', 'obb', 'classify']:
        predict_args["task"] = task

    # 创建predictions目录（确保输出目录存在）
    predictions_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    print(f"已创建预测输出目录: {predictions_dir}")

    # 开始预测
    print("=" * 60)
    print(f"开始对文件夹进行预测: {config['source']}")
    print(f"任务类型: {task}")
    print(f"预测参数: {predict_args}")
    print("=" * 60)
    
    try:
        results = model.predict(**predict_args)
        
        # 统计预测结果
        total_images = len(results) if isinstance(results, list) else 1
        
        # 统计检测结果（根据任务类型）
        total_detections = 0
        
        # 添加调试信息
        print(f"\n调试信息 - 检测结果结构:")
        for idx, r in enumerate(results[:3]):  # 只检查前3个结果
            print(f"  结果 {idx+1}:")
            attrs = ['boxes', 'obb', 'masks', 'keypoints']
            for attr in attrs:
                if hasattr(r, attr):
                    val = getattr(r, attr)
                    print(f"    {attr}: {type(val).__name__}, is None: {val is None}, len: {len(val) if hasattr(val, '__len__') else 'N/A'}")
        
        for r in results:
            # 对于不同的任务类型，检测结果可能在不同的属性中
            # 注意：OBB任务的结果可能在boxes中，也可能在obb中
            if hasattr(r, 'boxes') and r.boxes is not None:
                try:
                    count = len(r.boxes)
                    if count > 0:
                        total_detections += count
                except:
                    pass
            if hasattr(r, 'obb') and r.obb is not None:
                try:
                    count = len(r.obb)
                    if count > 0:
                        total_detections += count
                except:
                    pass
            if hasattr(r, 'masks') and r.masks is not None:
                try:
                    count = len(r.masks)
                    if count > 0:
                        total_detections += count
                except:
                    pass
            if hasattr(r, 'keypoints') and r.keypoints is not None:
                try:
                    count = len(r.keypoints)
                    if count > 0:
                        total_detections += count
                except:
                    pass
        
        print(f"\n预测完成！")
        print(f"处理图像数量: {total_images}")
        print(f"检测目标总数: {total_detections}")
        print(f"平均每张图像: {total_detections/total_images if total_images > 0 else 0:.2f} 个目标")
        
    except Exception as e:
        print(f"预测过程中出现错误: {e}")
        print("请检查配置参数是否正确")
        raise

    # 保存预测结果摘要
    results_summary = {
        "prediction_time": timestamp,
        "model_used": config.get('model', config['model_path']),
        "source": config['source'],
        "task": task,
        "image_size": config['img_size'],
        "batch_size": config['batch_size'],
        "device": config['device'],
        "conf_threshold": config.get('conf_threshold', 0.25),
        "iou_threshold": config.get('iou_threshold', 0.45),
        "max_det": config.get('max_det', 300),
        "total_images": total_images,
        "total_detections": total_detections,
        "output_dir": output_dir,
    }

    # 保存详细结果到JSON文件
    results_json_path = os.path.join(output_dir, 'prediction_summary.json')
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    # 保存结果摘要到文本文件
    summary_path = os.path.join(output_dir, 'prediction_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("YOLO11模型预测结果摘要\n")
        f.write("=" * 50 + "\n")
        f.write(f"预测时间: {timestamp}\n")
        f.write(f"使用模型: {config.get('model', config['model_path'])}\n")
        f.write(f"输入来源: {config['source']}\n")
        f.write(f"任务类型: {task}\n")
        f.write(f"图像尺寸: {config['img_size']}\n")
        f.write(f"批次大小: {config['batch_size']}\n")
        f.write(f"设备: {config['device']}\n")
        f.write(f"\n预测参数:\n")
        f.write(f"  置信度阈值: {config.get('conf_threshold', 0.25)}\n")
        f.write(f"  IoU阈值: {config.get('iou_threshold', 0.45)}\n")
        f.write(f"  最大检测数: {config.get('max_det', 300)}\n")
        f.write(f"\n预测结果:\n")
        f.write(f"  处理图像数: {total_images}\n")
        f.write(f"  检测目标总数: {total_detections}\n")
        f.write(f"  平均每张图像: {total_detections/total_images if total_images > 0 else 0:.2f} 个目标\n")
        f.write(f"\n输出目录: {output_dir}\n")

    print(f"\n预测完成。结果保存在: {output_dir}")
    
    # 检查输出目录中的文件
    predict_dir = os.path.join(output_dir, 'predictions')
    if os.path.exists(predict_dir):
        print(f"\n预测输出目录内容:")
        print(f"  {predict_dir}/")
        
        # 列出子目录
        for item in os.listdir(predict_dir):
            item_path = os.path.join(predict_dir, item)
            if os.path.isdir(item_path):
                file_count = len(os.listdir(item_path))
                print(f"    {item}/ ({file_count} 个文件)")
            else:
                print(f"    {item}")
    else:
        print(f"警告：预测输出目录不存在: {predict_dir}")
    
    print(f"\n预测结果摘要已保存到: {summary_path}")
    print(f"详细结果JSON已保存到: {results_json_path}")

    return results_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict with YOLO11 model on a folder of images")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file")
    args = parser.parse_args()

    config = load_config(args.config)
    predict_yolo11(config)
