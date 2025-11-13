import os
import shutil
from pathlib import Path

def extract_classes(yolo_root, class_ids, new_class_ids):
    """
    从YOLO数据集中提取指定类别的标注和图片，生成新的数据集。
    Args:
        yolo_root (str): YOLO数据集根目录
        class_ids (list): 需要提取的类别编号列表
        new_class_ids (list): 新类别编号列表（与class_ids一一对应）
    """
    class_map = {str(cid): str(nid) for cid, nid in zip(class_ids, new_class_ids)}
    for split in ['train', 'val']:
        label_dir = Path(yolo_root) / 'labels' / split
        image_dir = Path(yolo_root) / 'images' / split
        out_label_dir = Path(yolo_root) / 'labels_output' / split
        out_image_dir = Path(yolo_root) / 'images_output' / split
        out_label_dir.mkdir(parents=True, exist_ok=True)
        out_image_dir.mkdir(parents=True, exist_ok=True)

        for label_file in label_dir.glob('*.txt'):
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            selected_lines = []
            for line in lines:
                for cid, nid in class_map.items():
                    if line.strip().startswith(f'{cid} '):
                        new_line = nid + line[len(cid):]
                        selected_lines.append(new_line)
                        break
            if selected_lines:
                out_label_path = out_label_dir / label_file.name
                with open(out_label_path, 'w', encoding='utf-8') as f:
                    f.writelines(selected_lines)
                # 拷贝图片
                img_name = label_file.with_suffix('.jpg').name
                img_path = image_dir / img_name
                if not img_path.exists():
                    img_name = label_file.with_suffix('.png').name
                    img_path = image_dir / img_name
                if img_path.exists():
                    shutil.copy2(img_path, out_image_dir / img_name)
                else:
                    print(f'警告: 找不到图片 {img_name}，跳过。')

if __name__ == '__main__':
    yolo_root = r'C:\Users\Eugene\Desktop\vehicle_dataset\obb\all_yolo'
    extract_classes(yolo_root, class_ids=[0, 1, 2, 3], new_class_ids=[2])
    print('多类别数据提取完成！')
