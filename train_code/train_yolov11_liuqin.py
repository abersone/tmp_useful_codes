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
    exp_name = f"exp_{config['model_size']}_{config['img_size']}_{timestamp}"
    output_dir = os.path.join(config['output_base_dir'], timestamp, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # 保存使用的配置到输出目录
    config_output_path = os.path.join(output_dir, 'used_config.yaml')
    with open(config_output_path, 'w') as f:
        yaml.dump(config, f)

    # 初始化模型
    model = YOLO(config['model_path'])

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
        
        # "augment": True,  # 启用数据增强
        # "degrees": config['degrees'],
        # "scale": config['scale'],
        # "translate": config['translate'],
        # "fliplr": config['fliplr'],
        # "flipud": config['flipud'],
        # "mosaic": config['mosaic'],
        # "mixup": config['mixup'],
    }
    # if config['classes']:
    #     train_args['classes'] = config['classes']

    if config['cfg']:
        train_args['cfg'] = config['cfg']

    # 开始训练
    results = model.train(**train_args)

    # 保存训练结果摘要
    with open(os.path.join(output_dir, 'training_summary.txt'), 'w') as f:
        f.write(str(results))

    print(f"训练完成。结果保存在: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train yolo11 OBB model")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file")
    args = parser.parse_args()

    config = load_config(args.config)
    train_yolo11_obb(config)
