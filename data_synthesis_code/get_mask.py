import cv2
import numpy as np
import os
def generate_mask(template, cut_lines=20, std_ratio=0.5, debug=False, debug_dir=None):
    """
    生成模板图像的mask，通过裁切边缘来避免边缘像素的影响
    Args:
        template: 输入的模板图像
        cut_lines: 上下左右各裁切的像素行数
        debug: 是否输出中间过程图像
        debug_dir: 中间过程图像的保存路径,当该路径不为None且debug为True时保存
    Returns:
        mask: 二值化的mask图像，与输入template相同大小
    """
    # 转换为灰度图
    if len(template.shape) == 3:
        gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        gray = template
    if debug and debug_dir is not None:
        cv2.imwrite(os.path.join(debug_dir, "1_gray.png"), gray)
    
    h, w = gray.shape
    
    # 根据lines参数决定是否裁切边缘
    if cut_lines > 0:
        center = gray[cut_lines:-cut_lines, cut_lines:-cut_lines]
    else:
        center = gray
    if debug and debug_dir is not None:
        cv2.imwrite(os.path.join(debug_dir, "2_center.png"), center)
    
    # 高斯模糊
    blur = cv2.GaussianBlur(center, (3, 3), 0)
    if debug and debug_dir is not None:
        cv2.imwrite(os.path.join(debug_dir, "3_blur.png"), blur)
    
    # 计算均值和标准差
    mean, std = cv2.meanStdDev(blur)
    threshold = mean + std * std_ratio
    
    # 二值化
    _, binary = cv2.threshold(blur, threshold[0][0], 255, cv2.THRESH_BINARY)
    if debug and debug_dir is not None:
        cv2.imwrite(os.path.join(debug_dir, "4_binary.png"), binary)
    
    # 形态学操作
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # 开运算
    kernel5 = np.ones((5,5), np.uint8)
    binary = cv2.dilate(binary, kernel5)  # 膨胀
    binary = cv2.erode(binary, kernel5)   # 腐蚀
    if debug and debug_dir is not None:
        cv2.imwrite(os.path.join(debug_dir, "5_morphology.png"), binary)
    
    # 找到最大轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(gray)
    
    # 创建空白mask（与裁切后的尺寸相同）
    center_mask = np.zeros_like(binary)
    
    # 找到最大面积的轮廓
    max_contour = max(contours, key=cv2.contourArea)
    
    # 填充最大轮廓
    cv2.drawContours(center_mask, [max_contour], -1, 255, -1)
    if debug and debug_dir is not None:
        cv2.imwrite(os.path.join(debug_dir, "6_center_mask.png"), center_mask)
    
    # 创建与原图相同大小的mask
    if cut_lines > 0:
        full_mask = np.zeros_like(gray)
        full_mask[cut_lines:-cut_lines, cut_lines:-cut_lines] = center_mask
    else:
        full_mask = center_mask
    if debug and debug_dir is not None:
        cv2.imwrite(os.path.join(debug_dir, "7_full_mask.png"), full_mask)
    
    return full_mask

def process_template_folder(input_folder, output_folder, cut_lines=20, std_ratio=0.5, debug=False, debug_idx=0):
    """
    处理输入文件夹中的所有模板图片，生成对应的mask
    
    Args:
        input_folder: 输入文件夹路径,包含模板图片
        output_folder: 输出文件夹路径,用于保存生成的mask
        cut_lines: 裁切的边缘像素行数
        debug: 是否为debug模式
        debug_idx: debug模式下处理的图片序号
    """
    import os
    
    # 创建输出文件夹(如果不存在)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # 获取所有图片文件
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(input_folder) 
                  if os.path.splitext(f)[1].lower() in valid_extensions]
    
    print(f"在{input_folder}中找到{len(image_files)}个图片文件")
    
    # debug模式下检查序号是否有效
    if debug:
        if debug_idx >= len(image_files):
            print(f"错误: debug_idx {debug_idx} 超出图片数量范围 {len(image_files)}")
            return
        image_files = [image_files[debug_idx]]
        print(f"Debug模式: 仅处理第{debug_idx}个图片: {image_files[0]}")
    
    # 处理每个图片
    for image_file in image_files:
        # 读取图片
        image_path = os.path.join(input_folder, image_file)
        template = cv2.imread(image_path)
        
        if template is None:
            print(f"无法读取图片: {image_file}")
            continue
            
        # 生成mask
        mask = generate_mask(template, cut_lines=cut_lines, std_ratio=std_ratio, debug=debug, debug_dir=output_folder)
        
        # 构建输出文件名
        filename = os.path.splitext(image_file)[0]
        output_path = os.path.join(output_folder, f"{filename}_mask.png")
        
        # 保存mask
        cv2.imwrite(output_path, mask)
        print(f"已处理: {image_file} -> {output_path}")
        
    print("所有图片处理完成")

if __name__ == '__main__':
    # 示例使用
    input_folder = "/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/liuqin_data/dirt_tmpls/bg_light"  # 替换为实际的输入文件夹路径
    output_folder = "/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/liuqin_data/dirt_tmpls/bg_light_mask"    # 替换为实际的输出文件夹路径
    cut_lines = 1
    debug = True
    debug_idx = 95
    std_ratio = 1.0
    process_template_folder(input_folder, output_folder, cut_lines=cut_lines, std_ratio=std_ratio, debug=debug, debug_idx=debug_idx)