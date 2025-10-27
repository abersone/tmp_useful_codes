import os
import numpy as np
import cv2

#### 将原labelme2YOLO 后的N行5列+N行3列的瓶口.txt信息转换为N 行7列的.txt ####
def process_org_txt_file(input_images_folder, input_labels_folder, output_images_folder, output_labels_folder, pro_bbox_only = False):
    def adjust_rectangle_representation(rectangle):
        x1, y1, x2, y2 = rectangle
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        return center_x, center_y, width, height

    def point_inside_rectangle(point, rectangle):
        x, y = point
        x1, y1, x2, y2 = rectangle
        return x1 <= x <= x2 and y1 <= y <= y2
    
    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_labels_folder, exist_ok=True)
    
    for filename in os.listdir(input_images_folder):
        if filename.endswith(".png"):
            input_path = os.path.join(input_images_folder, filename)
            img = cv2.imread(input_path)

            # 写入新的.txt文件
            output_file_path = os.path.join(output_images_folder, os.path.basename(input_path))
            cv2.imwrite(output_file_path, img)
    
    for filename in os.listdir(input_labels_folder):
        print(filename)
        if filename.endswith(".txt"):
            input_path = os.path.join(input_labels_folder, filename)

            # 读取.txt文件
            with open(input_path, 'r') as file:
                lines = file.readlines()

            n = len(lines)

            rectangles = []
            image_points = []

            # 判断每一行的列数，分配给rectangles或image_points
            for line in lines:
                values = list(map(float, line.split()))
                if len(values) == 5:
                    rectangles.append(values[1:])
                elif len(values) == 3:
                    image_points.append(values[1:])

            if not pro_bbox_only:
                if len(rectangles) != len(image_points):
                    print("Error: the number of rectangles is not equal to image_points:", input_path)
                    break
            
            # 处理目标框的表示形式
            adjusted_rectangles = [adjust_rectangle_representation(rect) for rect in rectangles]

            # 创建N行7列的矩阵
            if pro_bbox_only:
                result_matrix = np.zeros((len(adjusted_rectangles), 5))
            else:
                result_matrix = np.zeros((len(adjusted_rectangles), 8))

            # 填充矩阵
            for i in range(len(adjusted_rectangles)):
                result_matrix[i, 0] = 0  # 类别
                result_matrix[i, 1:3] = adjusted_rectangles[i][:2]  # 矩形框中心坐标
                result_matrix[i, 3:5] = adjusted_rectangles[i][2:]  # 矩形框宽和高

                # 判断每个图像坐标点属于哪个目标框
                if not pro_bbox_only:
                    find_pt = False
                    for j, point in enumerate(image_points):
                        if point_inside_rectangle(point, rectangles[i]):
                            result_matrix[i, 5:7] = point  # 图像坐标点
                            result_matrix[i, 7] = 2
                            find_pt = True
                            break
                    if find_pt == False:
                        print("Error: image points are not matching to rectangles:", input_path)
                        return

            # 写入新的.txt文件
            output_file_path = os.path.join(output_labels_folder, os.path.basename(input_path))
            if pro_bbox_only:
                np.savetxt(output_file_path, result_matrix, fmt=['%d'] + ['%1.6f']*4, delimiter=' ')
            else:
                np.savetxt(output_file_path, result_matrix, fmt=['%d'] + ['%1.6f']*6 + ['%d'], delimiter=' ')
    print("Process end.")
                
                

#### 将N行7列的信息画在图像上 ####
def draw_boxes_and_points(input_images_folder, input_labels_folder, output_folder, is_gen_seg_image=False, postfix = ".png"):
    os.makedirs(output_folder, exist_ok=True)
    if is_gen_seg_image == True:
        output_folder_seg = os.path.join(output_folder, "seg_images")
        os.makedirs(output_folder_seg, exist_ok=True)
    index = 0
    for label_filename in os.listdir(input_labels_folder):
        index = index + 1
        print(str(index) + ": " + label_filename)
        if label_filename.endswith(".txt"):
            # 读取.txt文件
            label_filepath = os.path.join(input_labels_folder, label_filename)
            labels = np.loadtxt(label_filepath)
            if labels.size == 0:
                print("The label array is empty:", input_labels_folder)
                return
            if len(labels.shape) == 1:
                labels = [labels]
            # 读取对应的图像文件
            image_filename = os.path.splitext(label_filename)[0] + postfix
            image_filepath = os.path.join(input_images_folder, image_filename)
            image = cv2.imread(image_filepath)
            hei, wid = image.shape[0], image.shape[1]
            image_draw = image.copy()
            #print("0:", len(labels.shape))
            for idx, label in enumerate(labels):
                # print('1:', labels)
                # print("2:", label)
                # 提取目标框信息
                if label.shape[0] == 8:
                    category, center_x, center_y, box_width, box_height, point_x, point_y, _ = label.astype(float)
                    point_x = point_x * wid
                    point_y = point_y * hei
                elif label.shape[0] == 5:
                    category, center_x, center_y, box_width, box_height = label.astype(float)
                else:
                    continue
                center_x = center_x * wid
                center_y = center_y * hei
                box_width = box_width * wid
                box_height = box_height * hei

                # 画目标框
                color = (0, 0, 255) # tuple(np.random.randint(0, 255, 3).tolist())  # 随机颜色
                image_draw = cv2.rectangle(image_draw, (int(center_x - box_width / 2), int(center_y - box_height / 2)),
                              (int(center_x + box_width / 2), int(center_y + box_height / 2)), color, 2)

                # 画图像点
                if label.shape[0] == 8:
                    image_draw = cv2.circle(image_draw, (int(point_x), int(point_y)), 3, color, -1)

                # 裁切出3倍大小的rect image 分割用
                if is_gen_seg_image == True: 
                    box_width = box_width * 3
                    box_height = box_height * 3
                    bx0 = max(int(center_x - box_width / 2), 0)
                    by0 = max(int(center_y - box_height / 2), 0)
                    bx1 = min(int(center_x + box_width / 2), wid)
                    by1 = min(int(center_y + box_height / 2), hei)
                    
                    image_seg_filename = os.path.splitext(label_filename)[0] + "_rect" + str(idx+1) + postfix
                    output_seg_filepath = os.path.join(output_folder_seg, image_seg_filename)
                    image_crop = image[by0:by1, bx0:bx1]
                    cv2.imwrite(output_seg_filepath, image_crop)
                    
                    
            # 保存结果图像
            output_filepath = os.path.join(output_folder, image_filename)

            cv2.imwrite(output_filepath, image_draw)
            
    print("Drawing is finished.")       
    

# 遍历标签文件夹中的每个文件
def draw_seg_labels(images_folder, labels_folder, save_folder):
    def read_yolov8_labels(label_path, image_width, image_height):
        with open(label_path, 'r') as file:
            lines = file.readlines()
        
        bounding_boxes = []
        for line in lines:
            # 将每行的坐标字符串转换为浮点数列表
            data = [float(coord) for coord in line.strip().split()[1:]]
            
            # 将归一化坐标转换为图像坐标
            points = [(int(data[i] * image_width), int(data[i + 1] * image_height)) for i in range(0, len(data), 2)]
            
            bounding_boxes.append(points)
        
        return bounding_boxes

    def visualize_labels(image_path, label_path, save_path):
        # 读取图像
        image = cv2.imread(image_path)
        
        # 获取图像宽和高
        image_height, image_width, _ = image.shape
        
        # 读取YoloV8格式的标签
        bounding_boxes = read_yolov8_labels(label_path, image_width, image_height)
        
        # 在图像上绘制边界框
        for box in bounding_boxes:
            cv2.polylines(image, [np.array(box)], isClosed=True, color=(0, 255, 0), thickness=2)


        # 保存图像
        cv2.imwrite(save_path, image)
        
    ##
    os.makedirs(save_folder, exist_ok=True)
    for i, label_file in enumerate(os.listdir(labels_folder)):
        if label_file.endswith(".txt"):
            # 构建图像文件路径
            image_file = os.path.join(images_folder, label_file.replace(".txt", ".jpg"))
            
            # 构建标签文件路径
            label_file_path = os.path.join(labels_folder, label_file)
            
            # 可视化标签
            save_path = os.path.join(save_folder, label_file.replace(".txt", ".jpg"))
            visualize_labels(image_file, label_file_path, save_path)

def rename_png_files(folder_path):
    try:
        # 遍历文件夹中的所有文件
        for filename in os.listdir(folder_path):
            if filename.endswith('bmp'):
                # 提取前缀和后缀
                prefix, rest = filename[:-4], filename[-4:]
                print("prefix:", prefix)
                print('rest:', rest)
                # 将前缀的第一个字母移到最后
                new_filename = prefix[1:] + prefix[0] + rest

                print('new_filename:', new_filename)
                # 构建新的文件路径
                old_path = os.path.join(folder_path, filename)
                new_path = os.path.join(folder_path, new_filename)
                
                # 重命名文件
                os.rename(old_path, new_path)
                print(f'Renamed: {filename} to {new_filename}')
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# 1. 针对目标框+关键点的labelme 后数据转换  
# 2. 将目标框和关键点绘制在图像上
## pro_bbox_only = True 代表.json 文件只有目标框，没有关键点
## is_gen_seg_image=True 代表在绘制矩形框的时候，同步生成用于seg 用的裁切后的图像

## 首先，需使用labelme2yolo 将labelme 的数据转换成.txt（确保安装了labelme2yolo(github->git clone)）：
### 命令：$ python .\labelme2yolo.py --json_dir C:\Users\Eugene\Desktop\folder  --seg --val_size 0.2
if __name__ == "__main__":
    folder = "/home/liuqin/Desktop/projects/datasets/"
    is_pro_detect = False # True for Detection and False for Segmentation
    
    if is_pro_detect:
        input_folder_str = "/merry_scene_seletecd_labels"
        output_folder_str = "/merry_scene_seletecd_labels"
        pro_bbox_only = True # True: 标注时只有目标框；False：标注了目标框+关键点
        is_gen_seg_image = False
        # 1. 生成keypts datasets
        key_str = "/train"
        input_images_folder = folder + input_folder_str + "/images" + key_str
        input_labels_folder = folder + input_folder_str + "/labels" + key_str
        
        output_images_folder = folder + output_folder_str + "/images" + key_str
        output_labels_folder = folder + output_folder_str + "/labels" + key_str
        output_draw_folder = folder + output_folder_str + "/draw" + key_str
        # 处理每个.txt文件
        #process_org_txt_file(input_images_folder, input_labels_folder, output_images_folder, output_labels_folder, pro_bbox_only=pro_bbox_only)
        draw_boxes_and_points(output_images_folder, output_labels_folder, output_draw_folder, is_gen_seg_image=is_gen_seg_image, postfix=".jpg")
        key_str = "/val"
        input_images_folder = folder + input_folder_str + "/images" + key_str
        input_labels_folder = folder + input_folder_str + "/labels" + key_str
        
        output_images_folder = folder + output_folder_str + "/images" + key_str
        output_labels_folder = folder + output_folder_str + "/labels" + key_str
        output_draw_folder = folder + output_folder_str + "/draw" + key_str
        # 处理每个.txt文件
        #process_org_txt_file(input_images_folder, input_labels_folder, output_images_folder, output_labels_folder, pro_bbox_only=pro_bbox_only)
        draw_boxes_and_points(output_images_folder, output_labels_folder, output_draw_folder, is_gen_seg_image=is_gen_seg_image, postfix=".jpg")
        
        # key_str = "/test"
        # input_images_folder = folder + input_folder_str + "/images" + key_str
        # input_labels_folder = folder + input_folder_str + "/labels" + key_str
        
        # output_images_folder = folder + output_folder_str + "/images" + key_str
        # output_labels_folder = folder + output_folder_str + "/labels" + key_str
        # output_draw_folder = folder + output_folder_str + "/draw" + key_str
        # # 处理每个.txt文件
        # #process_org_txt_file(input_images_folder, input_labels_folder, output_images_folder, output_labels_folder, pro_bbox_only=pro_bbox_only)
        # draw_boxes_and_points(output_images_folder, output_labels_folder, output_draw_folder, is_gen_seg_image=is_gen_seg_image, postfix=".jpg")
    else:
        input_folder_str = "/lead_seg_v2.2"
        # 将Segmentation 的labels 画出来
        key_str = "/train"
        input_images_folder = folder + input_folder_str + "/images" + key_str
        input_labels_folder = folder + input_folder_str + "/labels" + key_str
        output_draw_folder = folder + input_folder_str + "/draw" + key_str
        draw_seg_labels(input_images_folder, input_labels_folder, output_draw_folder)
        
        key_str = "/val"
        input_images_folder = folder + input_folder_str + "/images" + key_str
        input_labels_folder = folder + input_folder_str + "/labels" + key_str
        output_draw_folder = folder + input_folder_str + "/draw" + key_str
        draw_seg_labels(input_images_folder, input_labels_folder, output_draw_folder)
    
    # rename
    # folder = "C:/Users/Eugene/Desktop/6DOF_data/[capture]0307_data"
    # input_images_folder = folder + "/data"
    # rename_png_files(input_images_folder)