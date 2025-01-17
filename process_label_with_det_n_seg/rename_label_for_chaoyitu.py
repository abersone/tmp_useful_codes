import os

def rename_txt_files(folder_path):
    # 列出文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 只处理.txt文件
        if filename.endswith(".txt"):
            # 拆分文件名，按下划线分隔
            parts = filename.split('_png')
            if len(parts) >= 1:  # 确保文件名包含足够的下划线
                # 新的文件名，保留前两部分加上.txt后缀
                new_filename = f"{parts[0]}.txt"
                # 获取文件的完整路径
                old_file = os.path.join(folder_path, filename)
                new_file = os.path.join(folder_path, new_filename)
                # 重命名文件
                os.rename(old_file, new_file)
                print(f"Renamed '{old_file}' to '{new_file}'")

# 示例文件夹路径
folder_path = "C:/Users/Eugene/Desktop/6DOF_data/#6_detect_8mm_labeling/bottle_det_ori_8mm_add/labels/val"
rename_txt_files(folder_path)
