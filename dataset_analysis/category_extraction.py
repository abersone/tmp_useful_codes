import os
from pathlib import Path


def extract_classes(input_folder, output_folder, class_ids, new_class_ids):
    """
    从输入文件夹中提取指定类别的标注，生成新的标签文件。
    
    Args:
        input_folder (str): 包含YOLO格式标签文件的输入文件夹
        output_folder (str): 输出标签文件的文件夹
        class_ids (list): 需要提取的类别编号列表
        new_class_ids (list): 新类别编号列表（与class_ids一一对应）
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # 检查输入文件夹是否存在
    if not input_path.exists():
        print(f"错误: 输入文件夹不存在: {input_folder}")
        return
    
    # 创建输出文件夹
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 验证参数
    if len(class_ids) != len(new_class_ids):
        print(f"错误: class_ids 和 new_class_ids 长度不一致！")
        print(f"  class_ids 长度: {len(class_ids)}, 内容: {class_ids}")
        print(f"  new_class_ids 长度: {len(new_class_ids)}, 内容: {new_class_ids}")
        return
    
    # 创建类别映射字典
    class_map = {str(cid): str(nid) for cid, nid in zip(class_ids, new_class_ids)}
    
    # 按类别ID长度降序排序，优先匹配较长的ID（避免类别10被误匹配为类别1）
    sorted_class_map = sorted(class_map.items(), key=lambda x: len(x[0]), reverse=True)
    
    # 打印实际的映射关系（用于调试）
    print(f"类别映射表: {class_map}")
    
    # 获取所有txt文件
    txt_files = list(input_path.glob('*.txt'))
    
    if not txt_files:
        print(f"警告: 在 {input_folder} 中未找到txt文件")
        return
    
    print(f"找到 {len(txt_files)} 个标签文件")
    print(f"要提取的类别: {class_ids}")
    print(f"对应的新类别ID: {new_class_ids}")
    
    processed_count = 0
    skipped_count = 0
    total_objects = 0
    extracted_objects = 0
    
    # 处理每个标签文件
    for label_file in txt_files:
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            selected_lines = []
            original_count = 0
            
            for line in lines:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                
                original_count += 1
                total_objects += 1
                
                # 参考原始文件的逻辑：使用startswith精确匹配类别ID
                # 这样只提取class_map中指定的类别，其他类别会被自动过滤掉
                matched = False
                for cid, nid in sorted_class_map:
                    # 使用startswith确保精确匹配（如"1 "匹配类别1，但不匹配"10 "）
                    if line_stripped.startswith(f'{cid} '):
                        # 重新映射类别ID：替换行首的类别ID
                        new_line = nid + line_stripped[len(cid):]
                        selected_lines.append(new_line)
                        extracted_objects += 1
                        matched = True
                        break  # 找到匹配后跳出循环
                
                # 如果类别不在class_map中，则跳过这一行（不提取）
            
            # 如果提取到了目标类别，保存文件
            if selected_lines:
                out_label_path = output_path / label_file.name
                with open(out_label_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(selected_lines) + '\n')
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"已处理 {processed_count} 个文件...")
            else:
                skipped_count += 1
                # 可选：显示跳过的文件（如果原始文件有内容但没匹配到类别）
                if original_count > 0:
                    # 这里可以添加调试信息，但为了避免输出过多，只在调试模式下显示
                    pass
                
        except Exception as e:
            print(f"处理文件 {label_file.name} 时出错: {e}")
            skipped_count += 1
            continue
    
    print(f"\n处理完成！")
    print(f"成功处理: {processed_count} 个文件")
    if skipped_count > 0:
        print(f"跳过: {skipped_count} 个文件（未包含目标类别）")
    print(f"总对象数: {total_objects}")
    print(f"提取对象数: {extracted_objects}")
    print(f"输出文件夹: {output_folder}")


def main():
    """
    主函数：定义输入输出文件夹和类别映射
    """
    # 定义输入文件夹（包含YOLO格式的txt标签文件）
    input_folder = r"C:\Users\Eugene\Desktop\ref_label_for_roboflow"
    
    # 定义输出文件夹（保存提取后的标签文件）
    output_folder = input_folder + r"\extracted_labels"
    
    # 定义要提取的类别ID列表
    # 例如：原始数据有类别[0,1,2,3]，只想提取类别[1,2]，则设置为 [1, 2]
    # 注意：只有class_ids中列出的类别会被提取，其他类别会被自动过滤掉
    class_ids = [3]  # 只提取类别1和2，类别0和3会被过滤掉
    
    # 定义新的类别ID列表（与class_ids一一对应，长度必须相同！）
    # 这些新ID将替换原类别ID
    # 选项1：保持原类别ID不变，设置为 [1, 2]
    # 选项2：重新映射为连续ID，设置为 [0, 1]（将类别1映射为0，类别2映射为1）
    # 选项3：映射为相同ID，设置为 [2, 2]（将类别1和2都映射为类别2）
    new_class_ids = [3]  # 保持原ID不变（类别1保持为1，类别2保持为2）
    
    # 处理标签文件
    extract_classes(input_folder, output_folder, class_ids, new_class_ids)
    print('类别提取完成！')


if __name__ == '__main__':
    main()

