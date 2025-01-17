import os
import random
import shutil
import numpy as np
import cv2

# 检查.txt 里有几行
def check_txt_lines(folder_path, line_num = 8):
    # Make sure the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    # Get all .txt files in the folder
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    for txt_file in txt_files:
        file_path = os.path.join(folder_path, txt_file)

        # Read lines from the file
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        if len(lines) != line_num:
            print("file lines is error:", txt_file)

    print("Check completed.")

# 将Label2Yolo seg输出的.txt 转换为训练yolo pose 需要的.txt 格式
def process_txt_files(image_folder, label_folder, output_image_folder, output_label_folder, pro_bbox_only = False, add_visible = False):
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)
    # Make sure the folder exists
    if not os.path.exists(label_folder):
        print(f"The folder '{label_folder}' does not exist.")
        return

    # Get all .png image files in the 'images' folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    for image_file in image_files:
        # Build the paths for image and label files
        image_path = os.path.join(image_folder, image_file)
        output_image_path = os.path.join(output_image_folder, image_file)
        # Read the image
        img = cv2.imread(image_path)
        cv2.imwrite(output_image_path, img)
    
    # Get all .txt files in the folder
    txt_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]

    for txt_file in txt_files:
        file_path = os.path.join(label_folder, txt_file)
        output_file_path = os.path.join(output_label_folder, txt_file)
        # Read lines from the file
        # print(file_path)
        with open(file_path, 'r') as file:
            lines = file.readlines()
        # print('lines:', lines)
        # Write the processed data back to the file
        with open(output_file_path, 'w') as file:
            # Write the first line as it is
            # file.write(lines[0].strip())
            data = list(map(float,  lines[0].strip().split()))
            x1, y1, x2, y2 = data[1], data[2], data[3], data[4]
            
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            wid = x2 - x1
            hei = y2 - y1
            line0 = f"{0} {cx} {cy} {wid} {hei}"
            file.write(line0)

            # Process lines starting from the 2nd line
            if not pro_bbox_only:
                if add_visible:
                    processed_data = ' '.join([' '.join(line.strip().split()[1:]) + " 2" if idx < 13 else ' '.join(line.strip().split()[1:]) + " 1" for idx, line in enumerate(lines[1:])])
                else:
                    processed_data = ' '.join([' '.join(line.strip().split()[1:]) for line in lines[1:]])
                                    # Write the processed data on the same line
                file.write(' ' + processed_data)

    print("Processing completed.")
    
# 基于yolo 的.txt pose labels 画矩形框和关键点
def draw_boxes_and_points(image_folder, label_folder, output_folder, draw_bbox_only = False, add_visible = True):
    # Make sure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all .png image files in the 'images' folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    for image_file in image_files:
        # Build the paths for image and label files
        image_path = os.path.join(image_folder, image_file)
        label_file = image_file.replace('.png', '.txt')
        label_path = os.path.join(label_folder, label_file)

        # Read the image
        img = cv2.imread(image_path)

        # Read rectangle information
        with open(label_path, 'r') as label_file:
            datas = list(map(float, label_file.readline().split()))
            # print('datas lens:', len(datas))
            box_data = datas[1:5]
            # Draw rectangle
            box_cx, box_cy, box_w, box_h = [int(d * img.shape[1] if i % 2 == 0 else d * img.shape[0]) for i, d in enumerate(box_data)]
            cv2.rectangle(img, (box_cx - box_w // 2, box_cy - box_h // 2), (box_cx + box_w // 2, box_cy + box_h // 2), (0, 0, 255), 2)

            # Read and draw point information
            if not draw_bbox_only:
                kpts_data = datas[5:]
                if add_visible:
                    for i in range(0, len(kpts_data) // 3):
                        point_x = round(kpts_data[i * 3] * img.shape[1])
                        point_y = round(kpts_data[i * 3 + 1] * img.shape[0])
                        point_flag = kpts_data[i * 3 + 2]
                            
                        if point_flag == 2:
                            cv2.circle(img, (point_x, point_y), 6, (255, 255, 0), -1)
                        elif point_flag == 1:
                            cv2.circle(img, (point_x, point_y), 6, (0, 255, 0), -1)
                        cv2.putText(img, str(i+1), (point_x, point_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    for i in range(0, len(kpts_data) // 2):
                        point_x = round(kpts_data[i * 2] * img.shape[1])
                        point_y = round(kpts_data[i * 2 + 1] * img.shape[0])    
                        cv2.circle(img, (point_x, point_y), 6, (0, 255, 255), -1)
                        c

                    

        # Save the processed image
        output_path = os.path.join(output_folder, "draw_" + image_file)
        # print("Save draw_image:", output_path)
        cv2.imwrite(output_path, img)

    print(f"Drawing completed.")

# 将一堆.png 和.txt 划分为train 和val 集
def split_data(images_folder, labels_folder, val_ratio=0.3):
    # Make sure the output folders exist
    train_images_folder = os.path.join(images_folder, 'train')
    val_images_folder = os.path.join(images_folder, 'val')
    train_labels_folder = os.path.join(labels_folder, 'train')
    val_labels_folder = os.path.join(labels_folder, 'val')

    for folder in [train_images_folder, val_images_folder, train_labels_folder, val_labels_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Get all .png image files in the 'images' folder
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.png')]

    # Calculate the number of images for validation
    val_size = int(len(image_files) * (1 - val_ratio))

    # Randomly select images for validation
    val_images = random.sample(image_files, val_size)

    for image_file in image_files:
        # Build the paths for image and label files
        image_path = os.path.join(images_folder, image_file)
        label_file = image_file.replace('.png', '.txt')
        label_path = os.path.join(labels_folder, label_file)

        # Determine whether to move the files to train or val folders
        if image_file in val_images:
            shutil.move(image_path, os.path.join(val_images_folder, image_file))
            shutil.move(label_path, os.path.join(val_labels_folder, label_file))
        else:
            shutil.move(image_path, os.path.join(train_images_folder, image_file))
            shutil.move(label_path, os.path.join(train_labels_folder, label_file))

    print("Splitting completed.")

# 基于原始的yolo pose .txt labels 裁切出ROI 进行训练，同步生成裁切后的.txt labels
def crop_and_update_data(images_folder, labels_folder, output_images_folder, output_labels_folder, add_visible = False):
    # Make sure the output folders exist
    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_labels_folder, exist_ok=True)

    # Get all image files in the images folder
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.png')]

    for image_file in image_files:
        # Load the image
        image_path = os.path.join(images_folder, image_file)
        img = cv2.imread(image_path)
        img_height, img_width, _ = img.shape
        # print('img_size:(w, h) = ', img_width, img_height)
        # Read corresponding label file
        label_file_path = os.path.join(labels_folder, image_file.replace('.png', '.txt'))
        with open(label_file_path, 'r') as label_file:
            data = list(map(float, label_file.readline().split()))
            category, cx, cy, wid, hei, *keypoints = data

            # Scale the width and height by 1.1
            x1_org = int((cx - wid / 2) * img_width)
            y1_org = int((cy - hei / 2) * img_height)
            x2_org = int((cx + wid / 2) * img_width)
            y2_org = int((cy + hei / 2) * img_height)

            # Calculate the rectangle coordinates in the original image
            length = min(wid, hei)
            expand_wid = wid + length * 2.2 #wid * 1.1
            expand_hei = hei + length * 2.2 #hei * 1.1
            x1 = int((cx - expand_wid / 2) * img_width)
            y1 = int((cy - expand_hei / 2) * img_height)
            x2 = int((cx + expand_wid / 2) * img_width)
            y2 = int((cy + expand_hei / 2) * img_height)
            # print('1-x1, y1, x2, y2, wid/hei:', x1, y1, x2, y2, int(expand_wid * img_width), int(expand_hei * img_height))
            x1_backup = x1
            x2_backup = x2
            y1_backup = y1
            y2_backup = y2
            x_shift = 0
            y_shift = 0
            if x1 < 0:
                x_shift = x1
                x1 = 0
                x2 = min(int(expand_wid * img_width), img_width)
            if y1 < 0:
                y_shift = y1
                y1 = 0
                y2 = min(int(expand_hei * img_height), img_height)
            if x2 >= img_width:
                x_shift = x2 - img_width
                x1 = max(0, int(img_width - 1 - expand_wid * img_width))
                x2 = img_width
            if y2 >= img_height:
                y_shift = y2 - img_height
                y1 = max(0, int(img_height - 1 - expand_hei * img_height))
                y2 = img_height
                
            
            # print('2-x1, y1, x2, y2, shift:', x1, y1, x2, y2, x_shift, y_shift)
            # Crop the rectangle from the original image
            cropped_img = img[y1:y2, x1:x2]

            # Save the cropped image to the output folder
            output_image_path = os.path.join(output_images_folder, image_file)
            cv2.imwrite(output_image_path, cropped_img)

            # Update the label data
            if cropped_img.shape[1] == img_width:
                cx_new = ((x1_backup + x2_backup) / 2) / cropped_img.shape[1]
            else:
                cx_new = ((x1_backup + x2_backup) / 2 + x_shift - x1_backup) / cropped_img.shape[1]
            if cropped_img.shape[0] == img_height:
                cy_new = ((y1_backup + y2_backup) / 2) / cropped_img.shape[0]
            else:
                cy_new = ((y1_backup + y2_backup) / 2 + y_shift - y1_backup) / cropped_img.shape[0]

            wid_new = (x2_org - x1_org) / cropped_img.shape[1]
            hei_new = (y2_org - y1_org) / cropped_img.shape[0]

            # Update keypoints data
            if add_visible:
                keypoints_new = [(kp * img_width - x1) / cropped_img.shape[1] if i % 3 == 0 else (kp * img_height - y1) / cropped_img.shape[0] if i % 3 == 1 else int(kp) for i, kp in enumerate(keypoints)]
            else:
                keypoints_new = [(kp * img_width - x1) / cropped_img.shape[1] if i % 2 == 0 else (kp * img_height - y1) / cropped_img.shape[0] for i, kp in enumerate(keypoints)]
            
            new_data = [cx_new, cy_new, wid_new, hei_new] + keypoints_new[0:]

            # Save the updated label data to the output folder
            output_label_path = os.path.join(output_labels_folder, image_file.replace('.png', '.txt'))
            with open(output_label_path, 'w') as output_label_file:
                output_label_file.write(" ".join('0') + " ")
                output_label_file.write(" ".join(map(str, new_data)) + "\n")

    print("Update completed.")


################### run ####################
line_num = 17
folder = "C:\\Users\\Eugene\\Desktop\\project\\6dof\\data\\det_seg\\cup\\data\\folder_2"
input_folder = "YOLODataset_seg"
output_folder = "cup_yolo_folder_2"
add_visible = True
# 1. 使用Label2Yolo 将labelme 的数据转换成.txt 
# 命令：$ python .\labelme2yolo.py --json_dir C:\Users\Eugene\Desktop\project\6dof\data\det_seg\cup\data\labeling\folder_2  --seg --val_size 0.2

# 2. Reorganize document content
# key_str = "train"
# labels_folder = os.path.join(folder, "YOLODataset_seg\\labels", key_str)
# check_txt_lines(labels_folder, line_num=line_num)
# key_str = "val"
# labels_folder = os.path.join(folder, "YOLODataset_seg\\labels", key_str)
# check_txt_lines(labels_folder, line_num=line_num)



# 3. Process detect data & draw detect data
key_str = "train"
images_folder = os.path.join(folder, input_folder + "\\images", key_str)
labels_folder = os.path.join(folder, input_folder + "\\labels", key_str)

output_images_folder = os.path.join(folder, output_folder + "_detect\\images", key_str)
output_labels_folder = os.path.join(folder, output_folder + "_detect\\labels", key_str)
output_draw_folder = os.path.join(folder, output_folder + "_detect\\draw", key_str)

process_txt_files(images_folder, labels_folder, output_images_folder, output_labels_folder, pro_bbox_only=True)
draw_boxes_and_points(output_images_folder, output_labels_folder, output_draw_folder, draw_bbox_only=True)

key_str = "val"
images_folder = os.path.join(folder, input_folder + "\\images", key_str)
labels_folder = os.path.join(folder, input_folder + "\\labels", key_str)

output_images_folder = os.path.join(folder, output_folder + "_detect\\images", key_str)
output_labels_folder = os.path.join(folder, output_folder + "_detect\\labels", key_str)
output_draw_folder = os.path.join(folder, output_folder + "_detect\\draw", key_str)

process_txt_files(images_folder, labels_folder, output_images_folder, output_labels_folder, pro_bbox_only=True)
draw_boxes_and_points(output_images_folder, output_labels_folder, output_draw_folder, draw_bbox_only=True)


# 4. Process kpts data & draw kpts data
key_str = "train"
images_folder = os.path.join(folder, input_folder + "\\images", key_str)
labels_folder = os.path.join(folder, input_folder + "\\labels", key_str)

output_images_folder = os.path.join(folder, output_folder + "_kpts\\images", key_str)
output_labels_folder = os.path.join(folder, output_folder + "_kpts\\labels", key_str)

process_txt_files(images_folder, labels_folder, output_images_folder, output_labels_folder, pro_bbox_only=False, add_visible=add_visible)

output_images_folder_crop = os.path.join(folder, output_folder + "_kpts_crop\\images", key_str)
output_labels_folder_crop = os.path.join(folder, output_folder + "_kpts_crop\\labels", key_str)
crop_and_update_data(output_images_folder, output_labels_folder, output_images_folder_crop, output_labels_folder_crop, add_visible=add_visible)
output_draw_folder = os.path.join(folder, output_folder + "_kpts_crop\\draw", key_str)

draw_boxes_and_points(output_images_folder_crop, output_labels_folder_crop, output_draw_folder, draw_bbox_only=False, add_visible = add_visible)

key_str = "val"
images_folder = os.path.join(folder, input_folder + "\\images", key_str)
labels_folder = os.path.join(folder, input_folder + "\\labels", key_str)

output_images_folder = os.path.join(folder, output_folder + "_kpts\\images", key_str)
output_labels_folder = os.path.join(folder, output_folder + "_kpts\\labels", key_str)

process_txt_files(images_folder, labels_folder, output_images_folder, output_labels_folder, pro_bbox_only=False, add_visible=add_visible)


output_images_folder_crop = os.path.join(folder, output_folder + "_kpts_crop\\images", key_str)
output_labels_folder_crop = os.path.join(folder, output_folder + "_kpts_crop\\labels", key_str)
crop_and_update_data(output_images_folder, output_labels_folder, output_images_folder_crop, output_labels_folder_crop, add_visible=add_visible)
output_draw_folder = os.path.join(folder, output_folder + "_kpts_crop\\draw", key_str)

draw_boxes_and_points(output_images_folder_crop, output_labels_folder_crop, output_draw_folder, draw_bbox_only=False, add_visible = add_visible)



