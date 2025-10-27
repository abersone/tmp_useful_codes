import os
import random
import shutil
import numpy as np

# 将一堆.jpg 和.txt 随机划分为train 和val 集
def split_data(images_folder, labels_folder, val_ratio=0.2):
    # Make sure the output folders exist
    train_images_folder = os.path.join(images_folder, 'train')
    val_images_folder = os.path.join(images_folder, 'val')
    train_labels_folder = os.path.join(labels_folder, 'train')
    val_labels_folder = os.path.join(labels_folder, 'val')

    for folder in [train_images_folder, val_images_folder, train_labels_folder, val_labels_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Get all .png image files in the 'images' folder
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]

    # Calculate the number of images for validation
    val_size = int(len(image_files) * val_ratio)

    # Randomly select images for validation
    val_images = random.sample(image_files, val_size)
    index = 0
    for image_file in image_files:
        # Build the paths for image and label files
        image_path = os.path.join(images_folder, image_file)
        label_file = image_file.replace('.jpg', '.txt')
        label_path = os.path.join(labels_folder, label_file)
        index = index + 1
        print("index:", index)
        # Determine whether to move the files to train or val folders
        if image_file in val_images:
            shutil.move(image_path, os.path.join(val_images_folder, image_file))
            shutil.move(label_path, os.path.join(val_labels_folder, label_file))
        else:
            shutil.move(image_path, os.path.join(train_images_folder, image_file))
            shutil.move(label_path, os.path.join(train_labels_folder, label_file))

    print("Splitting completed.")
    
if __name__ == "__main__":
    folder = "/home/liuqin/Desktop/projects/datasets/wire_v3/all"

    image_path = folder + "/images"
    label_path = folder + "/labels"
    split_data(image_path, label_path, 0.2)