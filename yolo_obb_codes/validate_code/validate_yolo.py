import argparse
from ultralytics import YOLO
import yaml

def load_dataset_config(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def validate_yolo(dataset_yaml, model_path, imgsz, workers=0):
    # 加载数据集配置
    dataset_config = load_dataset_config(dataset_yaml)
    
    # 加载模型
    model = YOLO(model_path)
    
    # 执行验证
    results = model.val(data=dataset_yaml, imgsz=imgsz, workers=workers, conf = 0.1)
    
    # 输出验证结果
    print("\n验证结果:")
    print(f"mAP50-95: {results.box.map}")
    print(f"mAP50: {results.box.map50}")
    print(f"mAP75: {results.box.map75}")
    print(f"Precision: {results.box.p}")
    print(f"Recall: {results.box.r}")
    
    # 如果是分割模型，还可以输出分割相关的指标
    if hasattr(results, 'seg'):
        print("\n分割指标:")
        print(f"Segmentation mAP50-95: {results.seg.map}")
        print(f"Segmentation mAP50: {results.seg.map50}")
        print(f"Segmentation mAP75: {results.seg.map75}")
    
    # 如果是OBB模型，可以输出OBB相关的指标
    if hasattr(results, 'obb'):
        print("\nOBB指标:")
        print(f"OBB mAP50-95: {results.obb.map}")
        print(f"OBB mAP50: {results.obb.map50}")
        print(f"OBB mAP75: {results.obb.map75}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Validate YOLO model on a dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset.yaml file")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size for the model")
    
    args = parser.parse_args()
    
    validate_yolo(args.dataset, args.model, args.imgsz)

if __name__ == "__main__":
    main()