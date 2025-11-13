import numpy as np
import cv2

def visualize_annotations(image_file, txt_file, output_image=None):
    """
    在图像上可视化YOLO格式的标注信息，不同类别使用不同颜色
    
    Args:
        image_file (str): 原始图片路径
        txt_file (str): YOLO格式的标注文件路径
        output_image (str, optional): 输出图片的保存路径
    """
    # 读取图片
    img = cv2.imread(image_file)
    img_height, img_width = img.shape[:2]
    
    # 定义颜色（BGR格式）
    colors = {
        0: (255, 128, 255),    # ball: 红色
        1: (255, 0, 0),  # pad: 蓝色      
        2: (0, 255, 0), # wire: 绿色
        3: (0, 255, 255),    # lead: 黄色
    }
    
    # 读取标注文件
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 绘制每个标注
    for line in lines:
        data = line.strip().split()
        if not data:
            continue
            
        # 获取类别ID和坐标点
        class_id = int(data[0])
        points = np.array([float(x) for x in data[1:]]).reshape(-1, 2)
        
        # 将归一化坐标转换回像素坐标
        points[:, 0] = points[:, 0] * img_width
        points[:, 1] = points[:, 1] * img_height
        points = points.astype(np.int32)
        
        # 绘制多边形，使用对应类别的颜色
        color = colors.get(class_id % len(colors), (0, 255, 0))  # 如果类别ID超出颜色列表，循环使用颜色
        cv2.polylines(img, [points], True, color, 2)
    
    # 保存图片
    cv2.imwrite(output_image, img)
    print(f'可视化结果已保存至：{output_image}')

if __name__ == '__main__':
    # 设置文件路径
    folder = r'C:\Users\Eugene\Desktop\code\wire3d_detection\data\input\vehicle\20251107_34pcs\ref_data'
    image_file = f'{folder}/ref_light.png'  
    output_txt = f'{folder}/ref_label.txt'  
    output_image = f'{folder}/ref_light_visualized.png'  
    
    # 可视化标注
    visualize_annotations(image_file, output_txt, output_image)
