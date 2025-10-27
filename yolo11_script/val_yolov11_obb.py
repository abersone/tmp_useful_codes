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

def validate_yolo11_obb(config):
    """验证YOLO11 OBB（旋转框）模型"""
    # 创建带时间戳的验证输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    val_name = f"val_obb_{config['model_size']}_{config['img_size']}_{timestamp}"
    output_dir = os.path.join(config['output_base_dir'], timestamp, val_name)
    os.makedirs(output_dir, exist_ok=True)

    # 保存使用的配置到输出目录
    config_output_path = os.path.join(output_dir, 'used_config.yaml')
    with open(config_output_path, 'w') as f:
        yaml.dump(config, f)

    # 初始化模型
    if config.get('model'):
        model = YOLO(config['model'])  # 使用指定的模型
        print(f"验证使用指定模型: {config['model']}")
    else:
        model = YOLO(config['model_path'])  # 使用默认的OBB模型
        print(f"验证使用YOLO官方模型: {config['model_path']}")

    # 设置验证参数
    val_args = {
        "data": config['data_yaml'],
        "imgsz": config['img_size'],
        "batch": config['batch_size'],
        "device": config['device'],
        
        "project": output_dir,
        "name": "val",
        "exist_ok": True,
        
        # 验证特定参数
        "conf": config.get('conf_threshold', 0.001),  # 置信度阈值
        "iou": config.get('iou_threshold', 0.6),      # IoU阈值
        "max_det": config.get('max_det', 300),        # 每张图像最大检测数量（所有类别总和）
        
        # 保存选项
        "save_txt": config.get('save_txt', True),     # 保存预测结果为txt文件
        "save_conf": config.get('save_conf', True),   # 保存置信度
        "save_json": config.get('save_json', True),   # 保存为COCO格式JSON
        "save_crop": config.get('save_crop', False),  # 保存裁剪的预测框
        
        # 可视化选项
        "plots": config.get('plots', True),           # 生成混淆矩阵等图表
        "show": config.get('show', False),            # 显示结果
        "show_labels": config.get('show_labels', True),  # 显示标签
        "show_conf": config.get('show_conf', True),   # 显示置信度
        
        "task": "obb",  # 指定任务类型为旋转框检测
    }

    # 开始验证
    print("开始验证OBB模型...")
    print(f"验证参数: {val_args}")
    try:
        results = model.val(**val_args)
        
        # 打印调试信息
        print(f"\n验证结果对象属性:")
        print(f"  results.box: {hasattr(results, 'box')}")
        print(f"  results.obb: {hasattr(results, 'obb')}")
        
        if hasattr(results, 'box'):
            print(f"  box.map50: {hasattr(results.box, 'map50')}")
            print(f"  box.mp: {hasattr(results.box, 'mp')}")
            print(f"  box.mr: {hasattr(results.box, 'mr')}")
            
        if hasattr(results, 'obb'):
            print(f"  obb.map50: {hasattr(results.obb, 'map50')}")
            print(f"  obb.mp: {hasattr(results.obb, 'mp')}")
            print(f"  obb.mr: {hasattr(results.obb, 'mr')}")
            
    except Exception as e:
        print(f"验证过程中出现错误: {e}")
        print("请检查配置参数是否正确")
        raise

    # 保存验证结果摘要
    results_summary = {
        "validation_time": timestamp,
        "model_used": config.get('model', config['model_path']),
        "dataset": config['data_yaml'],
        "image_size": config['img_size'],
        "batch_size": config['batch_size'],
        "device": config['device'],
        "metrics": {
            "mAP50": float(results.box.map50) if hasattr(results.box, 'map50') else None,
            "mAP50-95": float(results.box.map) if hasattr(results.box, 'map') else None,
            "precision": float(results.box.mp) if hasattr(results.box, 'mp') else None,
            "recall": float(results.box.mr) if hasattr(results.box, 'mr') else None,
            "obb_mAP50": float(results.obb.map50) if hasattr(results.obb, 'map50') else None,
            "obb_mAP50-95": float(results.obb.map) if hasattr(results.obb, 'map') else None,
            "obb_precision": float(results.obb.mp) if hasattr(results.obb, 'mp') else None,
            "obb_recall": float(results.obb.mr) if hasattr(results.obb, 'mr') else None,
        }
    }

    # 保存详细结果到JSON文件
    results_json_path = os.path.join(output_dir, 'validation_results.json')
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    # 保存结果摘要到文本文件
    summary_path = os.path.join(output_dir, 'validation_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("YOLO11 OBB（旋转框检测）模型验证结果摘要\n")
        f.write("=" * 50 + "\n")
        f.write(f"验证时间: {timestamp}\n")
        f.write(f"使用模型: {config.get('model', config['model_path'])}\n")
        f.write(f"数据集: {config['data_yaml']}\n")
        f.write(f"图像尺寸: {config['img_size']}\n")
        f.write(f"批次大小: {config['batch_size']}\n")
        f.write(f"设备: {config['device']}\n")
        f.write("\n检测性能指标:\n")
        f.write(f"  mAP@0.5: {results_summary['metrics']['mAP50']:.4f}\n")
        f.write(f"  mAP@0.5:0.95: {results_summary['metrics']['mAP50-95']:.4f}\n")
        f.write(f"  精确率: {results_summary['metrics']['precision']:.4f}\n")
        f.write(f"  召回率: {results_summary['metrics']['recall']:.4f}\n")
        f.write("\nOBB旋转框性能指标:\n")
        f.write(f"  OBB mAP@0.5: {results_summary['metrics']['obb_mAP50']:.4f}\n")
        f.write(f"  OBB mAP@0.5:0.95: {results_summary['metrics']['obb_mAP50-95']:.4f}\n")
        f.write(f"  OBB精确率: {results_summary['metrics']['obb_precision']:.4f}\n")
        f.write(f"  OBB召回率: {results_summary['metrics']['obb_recall']:.4f}\n")

    print(f"OBB模型验证完成。结果保存在: {output_dir}")
    
    # 检查输出目录中的文件
    val_dir = os.path.join(output_dir, 'val')
    if os.path.exists(val_dir):
        print(f"\n验证输出目录内容:")
        for root, dirs, files in os.walk(val_dir):
            level = root.replace(val_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:10]:  # 只显示前10个文件
                print(f"{subindent}{file}")
            if len(files) > 10:
                print(f"{subindent}... 还有 {len(files) - 10} 个文件")
    else:
        print(f"警告：验证输出目录不存在: {val_dir}")
    
    print(f"\n检测性能指标:")
    print(f"  mAP@0.5: {results_summary['metrics']['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95: {results_summary['metrics']['mAP50-95']:.4f}")
    print(f"  精确率: {results_summary['metrics']['precision']:.4f}")
    print(f"  召回率: {results_summary['metrics']['recall']:.4f}")
    print(f"\nOBB旋转框性能指标:")
    print(f"  OBB mAP@0.5: {results_summary['metrics']['obb_mAP50']:.4f}")
    print(f"  OBB mAP@0.5:0.95: {results_summary['metrics']['obb_mAP50-95']:.4f}")
    print(f"  OBB精确率: {results_summary['metrics']['obb_precision']:.4f}")
    print(f"  OBB召回率: {results_summary['metrics']['obb_recall']:.4f}")

    return results_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate YOLO11 OBB (Oriented Bounding Box) model")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file")
    args = parser.parse_args()

    config = load_config(args.config)
    validate_yolo11_obb(config)
