import argparse
import yaml
import pandas as pd
from validate_yolo import validate_yolo

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def batch_validate(config_path):
    config = load_config(config_path)
    results = []

    for dataset in config['datasets']:
        for model in config['models']:
            print(f"\n验证任务: 数据集 {dataset}, 模型 {model['path']}")
            validation_result = validate_yolo(dataset, model['path'], model['imgsz'], workers=0)

            result = {
                'Dataset': dataset,
                'Model': model['path'],
                'Precision': validation_result.box.p,
                'Recall': validation_result.box.r,
                'mAP50': validation_result.box.map50,
                'mAP50-95': validation_result.box.map
            }

            # 添加分割指标（如果有）
            if hasattr(validation_result, 'seg'):
                result.update({
                    'Seg_mAP50': validation_result.seg.map50,
                    'Seg_mAP50-95': validation_result.seg.map
                })

            # 添加OBB指标（如果有）
            if hasattr(validation_result, 'obb'):
                result.update({
                    'OBB_mAP50': validation_result.obb.map50,
                    'OBB_mAP50-95': validation_result.obb.map
                })

            results.append(result)

    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(results)
    csv_path = config.get('output_csv', 'validation_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n结果已保存到 {csv_path}")

    return df

def main():
    parser = argparse.ArgumentParser(description="Batch validate YOLO models on datasets")
    parser.add_argument("--config", type=str, required=True, help="Path to batch validation config file")
    
    args = parser.parse_args()
    
    if __name__ == "__main__":
        batch_validate(args.config)

if __name__ == "__main__":
    main()
