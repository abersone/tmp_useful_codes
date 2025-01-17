import os
import numpy as np
import cv2
import shutil


def read_seg_labels(label_path, image_width, image_height):
    with open(label_path, 'r') as file:
        lines = file.readlines()
    
    bounding_kpts_float = []
    bounding_kpts_int = []
    for line in lines:
        # 将每行的坐标字符串转换为浮点数列表
        data = [float(coord) for coord in line.strip().split()[1:]]
        
        # 将归一化坐标转换为图像坐标
        points = [(data[i] * image_width, data[i + 1] * image_height) for i in range(0, len(data), 2)]
        bounding_kpts_float.append(points)
        
        points_int = [(int(data[i] * image_width), int(data[i + 1] * image_height)) for i in range(0, len(data), 2)]
        bounding_kpts_int.append(points_int)
    
    return bounding_kpts_float, bounding_kpts_int

def get_seg_kpts(image_path, label_path):
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape

    bounding_kpts_float, bounding_kpts_int = read_seg_labels(label_path, image_width, image_height)
    
    # image_draw = image.copy()
    # for box in bounding_kpts_int:
    #     cv2.polylines(image_draw, [np.array(box)], isClosed=True, color=(0, 255, 0), thickness=2)
    # cv2.imwrite("local_seg_draw_test.jpg", image_draw)
    
    return bounding_kpts_float

def extract_info(string):
        # 初始化前缀为空字符串
        prefix = ""
        i = 0

        # 从开头开始遍历，直到找到 "rect" 前的最后一个下划线为止
        while i < len(string) and string[i:i+4] != "rect":
            if string[i] == "_":
                prefix += "_"
            else:
                prefix += string[i]
            i += 1
        prefix = prefix[0:len(prefix)-1]
        # 提取 "rect" 和 "." 之间的部分
        start_index = i + len("rect")
        end_index = string.find(".")
        number_str = string[start_index:end_index]
        
        # 将字符串转换为整数
        number = int(number_str)
        
        return prefix, number

def get_top_left_point(label_det_path, index, wid, hei):
    # read txt
    with open(label_det_path, 'r') as file:
        lines = file.readlines()

    if len(lines) < index:
        return -1, -1
    
    line = lines[index-1]
    center_x, center_y, box_width, box_height = [float(coord) for coord in line.strip().split()[1:]]
    center_x = center_x * wid
    center_y = center_y * hei
    box_width = box_width * wid
    box_height = box_height * hei
    
    box_width = box_width * 3
    box_height = box_height * 3
    start_x = max(center_x - box_width / 2, 0)
    start_y = max(center_y - box_height / 2, 0)
    
    return start_x, start_y

def is_file_in_folder(file_name, folder_path):
    # 拼接文件完整路径
    file_path = os.path.join(folder_path, file_name)
    
    # 检查文件是否存在并且是一个文件
    if os.path.isfile(file_path):
        return True
    else:
        print('No file:', file_path)
        return False
      
# 遍历标签文件夹中的每个文件
def generate_full_seg_label(seg_images_folder, seg_labels_folder, det_images_folder, det_labels_folder):
    
    full_seg_kpts = {}
    label_files = os.listdir(seg_labels_folder)

    for i, label_file in enumerate(label_files):
        print("Process: ", i, ", ", label_file)
        if label_file.endswith(".txt"):
            # 构建seg 图像文件路径
            image_seg_path = os.path.join(seg_images_folder, label_file.replace(".txt", ".png"))
            label_seg_path = os.path.join(seg_labels_folder, label_file)
            # 获取seg label 中关键点坐标
            bounding_kpts_float = get_seg_kpts(image_seg_path, label_seg_path)
            
           
            # 获取detect data 对应的路径和索引
            prefix, index = extract_info(label_file)
            
            image_det_path = os.path.join(det_images_folder, prefix + ".png")
            label_det_path = os.path.join(det_labels_folder, prefix + ".txt")
            
            # read image
            if not is_file_in_folder(prefix + ".png", det_images_folder):
                continue
            if not is_file_in_folder(prefix + ".txt", det_labels_folder):
                continue
            
            image = cv2.imread(image_det_path)
            hei, wid = image.shape[0], image.shape[1]
            image_draw = image.copy()
            x0, y0 = get_top_left_point(label_det_path, index, wid, hei)  
            if x0 < 0 or y0 < 0:
                continue
            
            
            for seg_kpts in bounding_kpts_float:
                seg_kpts_full_normal = [((x + x0) / wid, (y + y0) / hei) for x, y in seg_kpts]
                # seg_kpts_full = [(int(x + x0), int(y + y0)) for x, y in seg_kpts]
                # cv2.polylines(image_draw, [np.array(seg_kpts_full)], isClosed=True, color=(0, 255, 0), thickness=2)
                # cv2.imwrite("full_seg_draw_test.jpg", image_draw)
              
                if prefix in full_seg_kpts:
                    full_seg_kpts[prefix].append(seg_kpts_full_normal)
                else:
                    full_seg_kpts[prefix] = [seg_kpts_full_normal]
                    
    # return : full_seg_kpts: {key: [[(x,y), (x,y), ...], [(x,y), (x,y), ...], ...]}
    return full_seg_kpts
     
def filter_and_save_labels(full_seg_kpts, det_images_folder, det_labels_folder, output_folder):
    folder_images= os.path.join(output_folder, "images")
    folder_labels = os.path.join(output_folder, "labels")
    os.makedirs(folder_images, exist_ok=True)
    os.makedirs(folder_labels, exist_ok=True)
    folder_images_train = os.path.join(folder_images, "train")
    folder_images_val = os.path.join(folder_images, "val")
    folder_labels_train = os.path.join(folder_labels, "train")
    folder_labels_val = os.path.join(folder_labels, "val")
    os.makedirs(folder_images_train, exist_ok=True)
    os.makedirs(folder_images_val, exist_ok=True)
    os.makedirs(folder_labels_train, exist_ok=True)
    os.makedirs(folder_labels_val, exist_ok=True)
    
    ind = 0
    for key, value in full_seg_kpts.items():
        if not is_file_in_folder(key + '.png', det_images_folder):
            continue
        obj_num = len(value)
        
        label_det_path = os.path.join(det_labels_folder, key + ".txt")
        with open(label_det_path, 'r') as file:
            lines = file.readlines()

        if len(lines) != obj_num:
            continue
        ind = ind + 1
        print('save txt:', key + ".txt, ", ind)
        # 获取图像路径
        image_input_path = os.path.join(det_images_folder, key + '.png')
     
        if ind < 122:
            label_path = os.path.join(folder_labels_train, key + ".txt")
            image_path = os.path.join(folder_images_train, key + '.png')
        else:
            label_path = os.path.join(folder_labels_val, key + ".txt")
            image_path = os.path.join(folder_images_val, key + '.png')
        
        shutil.copyfile(image_input_path, image_path)
        with open(label_path, 'w') as f:
            for seg_info in value:
                f.write("0 ")
                for point in seg_info:
                    f.write(f"{point[0]} {point[1]} ")
                f.write("\n")
        
        
     
def main():
    folder = "C:/Users/Eugene/Desktop/6DOF_data/full_segment_data_pro"      
    seg_images_folder = folder + "/seg_images"
    seg_labels_folder = folder + "/seg_labels"
    det_images_folder = folder + "/det_images"
    det_labels_folder = folder + "/det_labels"
    output_folder = folder + "/bottle_segmentation_full"
    full_seg_kpts = generate_full_seg_label(seg_images_folder, seg_labels_folder, det_images_folder, det_labels_folder)
    filter_and_save_labels(full_seg_kpts, det_images_folder, det_labels_folder, output_folder)

                

if __name__ == '__main__':
    main()
    


            
