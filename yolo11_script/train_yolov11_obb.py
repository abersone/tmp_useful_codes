import argparse
from ultralytics import YOLO
import yaml
import os
from datetime import datetime

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train_yolo11_obb(config):
    # 创建带时间戳的实验输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"obb_exp_{config['model_size']}_{config['img_size']}_{timestamp}"
    output_dir = os.path.join(config['output_base_dir'], timestamp, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # 保存使用的配置到输出目录
    config_output_path = os.path.join(output_dir, 'used_config.yaml')
    with open(config_output_path, 'w') as f:
        yaml.dump(config, f)

    # 初始化模型
    if config.get('model'):
        model = YOLO(config['model'])  # 使用指定的模型
        print(f"training use the specified model: {config['model']}")
    else:
        model = YOLO(config['model_path'])  # 使用默认的OBB模型
        print(f"training use the yolo official OBB model: {config['model_path']}")

    # 设置训练参数
    train_args = {
        "data": config['data_yaml'],
        "epochs": config['epochs'],
        "imgsz": config['img_size'],
        "batch": config['batch_size'],
        "device": config['device'],
        
        "project": output_dir,
        "name": "train",
        "exist_ok": True,
        "pretrained": config['pretrained'],
        "patience": config['early_stopping_patience'],
        
        # 预训练相关参数
        "resume": config.get('resume', False),  # 是否从中断处继续训练
        "freeze": config.get('freeze', None),   # 冻结层数
        
        # 优化器参数
        "optimizer": config.get('optimizer', 'auto'),  # 优化器类型
        "lr0": config.get('lr0', 0.01),               # 初始学习率
        # "lrf": config.get('lrf', 0.01),               # 最终学习率因子
        # "momentum": config.get('momentum', 0.937),    # SGD动量
        # "weight_decay": config.get('weight_decay', 0.0005),  # 权重衰减
        # "warmup_epochs": config.get('warmup_epochs', 3.0),   # 预热轮数
        # "warmup_momentum": config.get('warmup_momentum', 0.8),  # 预热动量
        # "warmup_bias_lr": config.get('warmup_bias_lr', 0.1),   # 预热偏置学习率
        
        # 数据增强参数
        "augment": config['augment'],  # 启用数据增强
        "degrees": config['degrees'],
        "scale": config['scale'],
        "translate": config['translate'],
        "fliplr": config['fliplr'],
        "flipud": config['flipud'],
        "mosaic": config['mosaic'],
        "mixup": config['mixup'],
        
        # 裁切相关参数
        "copy_paste": config['copy_paste'],
        "crop_fraction": config['crop_fraction'],
        
        "task": "obb",  # 指定任务类型为OBB
    }

    if config.get('cfg'):
        train_args['cfg'] = config['cfg']

    # 开始训练
    results = model.train(**train_args)

    # 保存训练结果摘要
    with open(os.path.join(output_dir, 'training_summary.txt'), 'w') as f:
        f.write(str(results))

    print(f"OBB模型训练完成。结果保存在: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO11 OBB model")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file")
    args = parser.parse_args()

    config = load_config(args.config)
    train_yolo11_obb(config) 