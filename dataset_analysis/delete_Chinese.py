import os
import re

def remove_chinese_from_filename(folder_path):
    """
    删除文件夹中所有文件名中的中文字符
    
    Args:
        folder_path (str): 文件夹路径
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在")
        return
    
    # 中文字符的正则表达式模式
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 跳过文件夹，只处理文件
        if os.path.isfile(file_path):
            # 获取文件名和扩展名
            name, ext = os.path.splitext(filename)
            
            # 删除中文字符
            new_name = chinese_pattern.sub('', name)
            
            # 如果删除中文字符后文件名为空，添加默认名称
            if not new_name.strip():
                new_name = "file"
            
            # 构建新的文件名
            new_filename = new_name + ext
            new_file_path = os.path.join(folder_path, new_filename)
            
            # 如果新文件名与原文件名不同，则重命名
            if new_filename != filename:
                try:
                    # 处理重名文件
                    counter = 1
                    original_new_file_path = new_file_path
                    while os.path.exists(new_file_path):
                        new_filename = f"{new_name}_{counter}{ext}"
                        new_file_path = os.path.join(folder_path, new_filename)
                        counter += 1
                    
                    os.rename(file_path, new_file_path)
                    print(f"重命名: '{filename}' -> '{new_filename}'")
                except Exception as e:
                    print(f"重命名文件 '{filename}' 时出错: {e}")

def main():
    """主函数"""
    # 获取用户输入的文件夹路径
    folder = r"C:\Users\Eugene\Desktop\2d_wire_product\BK(P-0805)\X65"
    folder_path = folder + r"\light"
    
    # 去除路径两端的引号（如果有的话）
    folder_path = folder_path.strip('"\'')
    
    print(f"开始处理文件夹: {folder_path}")
    print("=" * 50)
    
    # 执行重命名操作
    remove_chinese_from_filename(folder_path)
    
    print("=" * 50)
    print("处理完成！")
    
    folder_path = folder + r"\dark"
    
    # 去除路径两端的引号（如果有的话）
    folder_path = folder_path.strip('"\'')
    
    print(f"开始处理文件夹: {folder_path}")
    print("=" * 50)
    
    # 执行重命名操作
    remove_chinese_from_filename(folder_path)
    
    print("=" * 50)
    print("处理完成！")

if __name__ == "__main__":
    main()