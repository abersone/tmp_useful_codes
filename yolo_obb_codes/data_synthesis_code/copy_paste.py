import cv2
import numpy as np
from skimage.exposure import match_histograms
import os
# 将一个灰度图像块采用泊松融合到灰度图像指定位置上
def poisson_blend(image, block, loc=(0, 0), mask=None):  # loc: (x, y)
    # 检查图像是否为灰度图像
    if len(image.shape)!= 3:
        raise ValueError('image must be a color image')
    if len(block.shape)!= 3:
        raise ValueError('block must be a color image')
    # 检查图像块block的尺寸必须大于0
    if block.shape[0] <= 0 or block.shape[1] <= 0:
        raise ValueError('block must be larger than 0')
    # 检查图像块block的尺寸必须小于image的尺寸
    if block.shape[0] >= image.shape[0] or block.shape[1] >= image.shape[1]:
        raise ValueError('block must be smaller than image')

    # 如果block在图像loc的位置出现越界，将block裁剪到图像边界
    if loc[0] < block.shape[1] // 2:
        loc = (block.shape[1] // 2, loc[1])
    if loc[1] < block.shape[0] // 2:
        loc = (loc[0], block.shape[0] // 2)
    if loc[0] >= image.shape[1] - block.shape[1] // 2:
        loc = (image.shape[1] - block.shape[1] // 2, loc[1])
    if loc[1] >= image.shape[0] - block.shape[0] // 2:
        loc = (loc[0], image.shape[0] - block.shape[0] // 2)

    # 处理mask
    if mask is None:
        mask = 255 * np.ones(block.shape[:2], dtype=np.uint8)
    else:
        # 确保mask是单通道
        if len(mask.shape) > 2:
            mask = mask[:,:,0]
        
        # 将mask二值化，注意这里可能需要反转mask
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY) 
        
        # 如果mask中的目标区域是255（白色），则不需要反转
        # 如果发现融合效果不好，可以取消下面这行的注释来反转mask
        # mask = cv2.bitwise_not(mask)
    
    # 确保mask与block尺寸一致
    if mask.shape[:2] != block.shape[:2]:
        raise ValueError('Mask must have the same dimensions as block')

    # 保存处理后的mask图像用于调试
    cv2.imwrite('processed_mask.jpg', mask)
    
    # 使用MIXED_CLONE模式进行融合
    # cv2.NORMAL_CLONE: 普通融合模式
    # cv2.MIXED_CLONE: 混合融合模式
    # cv2.MONOCHROME_TRANSFER: 仅转移源图像的亮度模式
    dst = cv2.seamlessClone(block, image, mask, loc, cv2.MIXED_CLONE) 
    return dst

def blend_template(template, mask, target_img, pos, blur_size=5, opacity=0.95, dilate_size=5,save_process=True, debug_dir=None):
    """
    将模板图像通过alpha混合的方式融合到目标图像的指定位置
    
    Args:
        template: 模板图像，BGR格式
        mask: 二值mask，单通道，255表示前景
        target_img: 目标图像，BGR格式
        pos: 模板左上角在目标图像中的位置，tuple (x, y)
        blur_size: 高斯模糊的核大小，必须是奇数
        opacity: 整体透明度，范围0-1，值越小越透明
        save_process: 是否保存中间结果
        debug_dir: 中间过程图像的保存路径,当该路径不为None且save_process为True时保存
    """
    # print("开始图像融合...")
    # 如果debug_dir不为None且不存在,则创建该目录
    if debug_dir is not None and not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
        print(f"创建调试目录: {debug_dir}")
        
    # 确保输入图像格式正确
    if len(template.shape) != 3:
        raise ValueError("Template must be a BGR image")
    if len(mask.shape) != 2:
        raise ValueError("Mask must be a single channel image")
    if len(target_img.shape) != 3:
        raise ValueError("Target image must be a BGR image")
        
    # 复制目标图像，避免修改原图
    result_img = target_img.copy()
    h, w = template.shape[:2]
    x, y = pos
    # print(f"模板尺寸: {w}x{h}, 目标位置: ({x}, {y})")
    
    # 检查位置是否有效
    if x < 0 or y < 0 or x + w > target_img.shape[1] or y + h > target_img.shape[0]:
        raise ValueError("Invalid position: template would be outside target image")
    
    # 检查模板和mask尺寸是否匹配
    if template.shape[:2] != mask.shape[:2]:
        raise ValueError("Template and mask must have the same dimensions")
    
    # 对mask进行膨胀操作
    if (dilate_size > 0 and dilate_size % 2 == 1):
        kernel_size = dilate_size  # 可以根据需要调整核的大小
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)  # iterations控制膨胀次数
    
    if save_process and debug_dir is not None:
        cv2.imwrite(os.path.join(debug_dir, '00_dilated_mask.jpg'), mask)
        
    # 创建alpha通道（将mask转换为0-1范围的float类型）并调整整体透明度
    alpha = mask.astype(np.float32) / 255.0
    alpha = alpha * opacity  # 调整整体透明度
    
    # 对alpha通道进行高斯模糊，使边缘更自然
    blur_size = blur_size + (1 - blur_size % 2)  # 确保是奇数
    alpha = cv2.GaussianBlur(alpha, (blur_size, blur_size), 0)
    
    # 扩展alpha为3通道
    alpha = np.stack([alpha] * 3, axis=2)
    
    # 提取目标区域
    roi = result_img[y:y+h, x:x+w]
    
    # 直接进行alpha混合
    result = alpha * template + (1 - alpha) * roi
    
    # 保存中间结果（如果需要）
    if save_process and debug_dir is not None:
        cv2.imwrite(os.path.join(debug_dir, '01_template.jpg'), template)
        cv2.imwrite(os.path.join(debug_dir, '02_roi.jpg'), roi)
        cv2.imwrite(os.path.join(debug_dir, '03_alpha_mask.jpg'), (alpha[:,:,0] * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(debug_dir, '04_blend_result.jpg'), result)
    
    # 将结果放回原图
    result_img[y:y+h, x:x+w] = result
    
    # print("图像融合完成")
    return result_img.astype(np.uint8)

def test_blend():
    """
    测试函数
    """
    # 加载图像
    template = cv2.imread("/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/liuqin_data/dirt_tmpls/real_dirt/1412168_17_56_32_934_crop_0.jpg")
    mask = cv2.imread("/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/liuqin_data/dirt_tmpls/mask/1412168_17_56_32_934_crop_0_mask.png", 0)  # 以灰度图方式读取mask
    target = cv2.imread('/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/liuqin_data/fail_data/images/train/1437788_10_51_46_146_crop.jpg')
    output_folder = "/mnt/nfs/AOI_detection/ccm/data/v6/after_screening/liuqin_data/data_synthesis_code"
    # 测试Alpha融合
    result = blend_template(
        template=template,
        mask=mask,
        target_img=target,
        pos=(420, 483), #(420, 483),  # (1077, 534),  # 示例位置(420, 483)
        blur_size=30,
        opacity=0.95,
        dilate_size=5,
        save_process=True,
        debug_dir=output_folder
    )
    
    # 保存结果
    cv2.imwrite(os.path.join(output_folder, 'blend_result.jpg'), result)
    
    # 测试泊松融合
    # 检查mask的值范围
    # mask_min = mask.min()
    # mask_max = mask.max()
    # print(f"mask的最小值: {mask_min}")
    # print(f"mask的最大值: {mask_max}")

    # result = poisson_blend(target, template, loc=(1470, 528), mask=mask) # 示例位置
    
    # # 保存结果
    # cv2.imwrite(os.path.join(output_folder, 'poisson_result.jpg'), result)

if __name__ == '__main__':
    test_blend()