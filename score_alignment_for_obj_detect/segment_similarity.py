import os
import cv2
import numpy as np

######## 1. 基于mask label 生成mask 图像 #########
def read_segmentation_labels(label_file):
    label_list = []
    with open(label_file, 'r') as file:
        lines = file.readlines()      
        for line in lines:
            data = line.strip().split()
            label_list.append(data) 
    return label_list

def generate_mask_image(label_data, image_width, image_height):
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    for data in label_data:
        points = [(int(float(data[i]) * image_width), int(float(data[i+1]) * image_height)) for i in range(1, len(data), 2)]
        cv2.fillPoly(mask, [np.array(points)], color=255)
    return mask

def generate_label_mask(image_folder, label_folder, output_folder):
    image_files = os.listdir(image_folder)
    label_files = os.listdir(label_folder)
    os.makedirs(output_folder, exist_ok=True)

    for label_file in label_files:
        print(label_file)
        if label_file.endswith('.txt'):
            label_name = os.path.splitext(label_file)[0]
            img_file = label_name + '.png'
            if img_file in image_files:
                image_path = os.path.join(image_folder, label_name + '.png')
                image = cv2.imread(image_path)
                image_height, image_width, _ = image.shape
                
                label_list = read_segmentation_labels(os.path.join(label_folder, label_file))
                
                mask = np.zeros((image_height, image_width), dtype=np.uint8)
                for idx, data in enumerate(label_list):
                    points = [(int(float(data[i]) * image_width), int(float(data[i+1]) * image_height)) for i in range(1, len(data), 2)]
                    cv2.fillPoly(mask, [np.array(points)], color=255)
                    # save
                    mask_name = label_name + '_' + str(idx) + '_mask.jpg'
                    mask_path = os.path.join(output_folder, mask_name)
                    cv2.imwrite(mask_path, mask)

######## 2. 计算相似度 #########  
def calculate_similarity(image1, image2):
    # 确保两幅图像具有相同的大小
    assert image1.shape == image2.shape, "两幅图像大小不一致"

    # 将图像中的非零值统一为1
    image1 = (image1 != 0).astype(int)
    image2 = (image2 != 0).astype(int)

    # 计算相同像素点数
    same_pixels = (image1 == image2).sum()

    # 计算总像素数
    total_pixels = image1.size

    # 计算相似度百分比
    similarity = (1.0 * same_pixels / total_pixels) * 100.0

    return similarity

# 读取单张图像
def read_image_from_folder(folder, filename):
    image_path = os.path.join(folder, filename)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

def main():
    # 1. 生成Label的mask图
    # image_folder = 'data/generate_mask/segment_verify_image'
    # label_folder = 'data/generate_mask/segment_verify_label'
    # output_folder ='data/generate_mask/bottle_seg_label'
    # generate_label_mask(image_folder, label_folder, output_folder)
    # return
    # 2. 计算相似度
    folder1_path = 'align_score/qnn_segment_mask_0516'
    folder2_path = 'align_score/bottle_seg_label'

    # 获取两个文件夹中的文件名列表，并取交集
    folder1_files = set(os.listdir(folder1_path))
    folder2_files = set(os.listdir(folder2_path))
    common_files = folder1_files.intersection(folder2_files)
    print("mask numbers:", len(common_files))

    # 初始化变量
    total_similarity = 0
    num_images = 0

    # 逐一处理每对图像
    similar_max = 0
    similar_min = 1000
    for filename in common_files:
        # 读取图像
        image1 = read_image_from_folder(folder1_path, filename)
        image2 = read_image_from_folder(folder2_path, filename)
        
        # 计算相似度
        similarity = calculate_similarity(image1, image2)
        if similarity > similar_max:
            similar_max = similarity
        if similarity < similar_min:
            similar_min = similarity
        # 更新总相似度和图像对数
        total_similarity += similarity
        num_images += 1

    # 计算相似度平均值
    average_similarity = total_similarity / num_images
    print(folder1_path)
    print("average similarity ratio: {:.2f}%".format(average_similarity))
    print("max similarity ratio: {:.2f}%".format(similar_max))
    print("min similarity ratio: {:.2f}%".format(similar_min))
    
if __name__ == '__main__':
    main()
    
