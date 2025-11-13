import os
from pathlib import Path


def combine_labels(folder1, folder2, output_folder):
    """
    合并两个文件夹中相同名称的YOLO11分割标签文件
    
    Args:
        folder1 (str): 第一个标签文件夹路径
        folder2 (str): 第二个标签文件夹路径
        output_folder (str): 输出文件夹路径
    """
    folder1_path = Path(folder1)
    folder2_path = Path(folder2)
    output_path = Path(output_folder)
    
    # 检查输入文件夹是否存在
    if not folder1_path.exists():
        print(f"错误: 第一个文件夹不存在: {folder1}")
        return
    
    if not folder2_path.exists():
        print(f"错误: 第二个文件夹不存在: {folder2}")
        return
    
    # 创建输出文件夹
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取两个文件夹中的所有txt文件
    files1 = {f.stem: f for f in folder1_path.glob('*.txt')}
    files2 = {f.stem: f for f in folder2_path.glob('*.txt')}
    
    # 找到两个文件夹中共同的文件名
    common_files = set(files1.keys()) & set(files2.keys())
    
    # 找到只在某个文件夹中存在的文件
    only_in_folder1 = set(files1.keys()) - set(files2.keys())
    only_in_folder2 = set(files2.keys()) - set(files1.keys())
    
    if not common_files:
        print("警告: 未找到两个文件夹中共同的文件")
        if only_in_folder1:
            print(f"  仅在文件夹1中的文件数: {len(only_in_folder1)}")
        if only_in_folder2:
            print(f"  仅在文件夹2中的文件数: {len(only_in_folder2)}")
        return
    
    print(f"文件夹1中找到 {len(files1)} 个txt文件")
    print(f"文件夹2中找到 {len(files2)} 个txt文件")
    print(f"找到 {len(common_files)} 个共同的文件")
    if only_in_folder1:
        print(f"仅在文件夹1中的文件数: {len(only_in_folder1)}")
    if only_in_folder2:
        print(f"仅在文件夹2中的文件数: {len(only_in_folder2)}")
    
    processed_count = 0
    skipped_count = 0
    total_lines_folder1 = 0
    total_lines_folder2 = 0
    total_lines_output = 0
    
    # 处理每个共同的文件
    for filename_stem in common_files:
        try:
            file1_path = files1[filename_stem]
            file2_path = files2[filename_stem]
            
            # 读取第一个文件的内容
            with open(file1_path, 'r', encoding='utf-8') as f:
                lines1 = [line.strip() for line in f.readlines() if line.strip()]
            
            # 读取第二个文件的内容
            with open(file2_path, 'r', encoding='utf-8') as f:
                lines2 = [line.strip() for line in f.readlines() if line.strip()]
            
            # 合并内容：先写入文件夹1的内容，再写入文件夹2的内容
            combined_lines = lines1 + lines2
            
            # 写入输出文件
            output_file = output_path / f"{filename_stem}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(combined_lines))
                if combined_lines:  # 如果不是空文件，添加换行符
                    f.write('\n')
            
            processed_count += 1
            total_lines_folder1 += len(lines1)
            total_lines_folder2 += len(lines2)
            total_lines_output += len(combined_lines)
            
            if processed_count % 100 == 0:
                print(f"已处理 {processed_count} 个文件...")
                
        except Exception as e:
            print(f"处理文件 {filename_stem}.txt 时出错: {e}")
            skipped_count += 1
            continue
    
    # 打印统计信息
    print(f"\n处理完成！")
    print(f"成功合并: {processed_count} 个文件")
    if skipped_count > 0:
        print(f"跳过: {skipped_count} 个文件")
    print(f"文件夹1总行数: {total_lines_folder1}")
    print(f"文件夹2总行数: {total_lines_folder2}")
    print(f"合并后总行数: {total_lines_output}")
    print(f"输出文件夹: {output_folder}")


def main():
    """
    主函数：定义两个输入文件夹和输出文件夹路径
    """
    # 定义第一个标签文件夹路径
    folder1 = r"C:\Users\Eugene\Desktop\vehicle_dataset\obb\all82_image\wire"
    
    # 定义第二个标签文件夹路径
    folder2 = r"C:\Users\Eugene\Desktop\vehicle_dataset\obb\all82_image\no_wire"
    
    # 定义输出文件夹路径
    output_folder = r"C:\Users\Eugene\Desktop\vehicle_dataset\obb\all82_image\global_seg_all"
    
    # 合并标签文件
    combine_labels(folder1, folder2, output_folder)
    print('标签合并完成！')


if __name__ == '__main__':
    main()

