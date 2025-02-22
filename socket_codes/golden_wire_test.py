import numpy as np
import matplotlib
matplotlib.use('Agg')  # 在导入pyplot之前设置
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from pathlib import Path
from sklearn.cluster import DBSCAN
import cv2
from scipy import optimize
from depth_to_pointcloud import DepthToPointCloud

def process_single_frame(ini_path, tiff_path, png_path, roi_vertices, output_prefix='output', b_save_pointcloud=False):
    # 创建转换器实例
    converter = DepthToPointCloud(vertices=roi_vertices)
    converter.read_config(ini_path)
    depth_image = converter.read_depth_image(tiff_path)
    # 可视化深度图
    plt.figure(figsize=(10, 8))
    plt.imshow(depth_image, cmap='jet')
    plt.colorbar()
    plt.title('Depth Map')
    plt.savefig(f'{output_prefix}_depth_image.png')
    plt.close()
    # 处理原始RGB图像
    cropped_image, bbox = converter.crop_roi_from_image(png_path, output_prefix)
    
    # 转换为点云
    points_3d, points_2d = converter.convert_to_pointcloud(depth_image, bbox)
    
    # 根据开关决定是否保存点云
    if b_save_pointcloud:
        pointcloud_3d_path = f'{output_prefix}_pointcloud_3d.txt'
        pointcloud_2d_path = f'{output_prefix}_pointcloud_2d.txt'
        converter.save_pointcloud(points_3d, points_2d, pointcloud_3d_path, pointcloud_2d_path)
        print(f"已将点云保存至 {pointcloud_3d_path} 和 {pointcloud_2d_path}")
        
    return points_3d, points_2d

def process_folder(folder_path, output_folder, roi_vertices, start_idx=1, end_idx=50, b_save_pointcloud=False):
    """
    处理指定文件夹中的所有数据
    
    参数:
    folder_path: 数据文件夹路径
    roi_vertices: ROI顶点坐标
    start_idx: 起始索引（默认1）
    end_idx: 结束索引（默认50）
    b_save_pointcloud: 是否保存点云
    """
    # 确保文件夹路径是字符串类型
    folder_path = str(Path(folder_path))
    
    # 创建输出文件夹
    output_folder.mkdir(exist_ok=True)
    
    # 处理每一帧数据
    for idx in range(start_idx, end_idx + 1):
        print(f"\n处理第 {idx} 帧...")
        
        # 构建文件路径
        ini_path = str(Path(folder_path) / f"{idx}.ini")
        tiff_path = str(Path(folder_path) / f"{idx}.tiff")
        png_path = str(Path(folder_path) / f"{idx}_0.png")
        
        # 检查文件是否存在
        if not all(Path(f).exists() for f in [ini_path, tiff_path, png_path]):
            print(f"警告: 第 {idx} 帧的某些文件不存在，跳过处理")
            continue
            
        try:
            # 设置输出前缀（包含完整路径）
            output_prefix = str(output_folder / f"frame_{idx}")
            
            # 处理单帧数据
            points3d, points2d = process_single_frame(
                ini_path=ini_path,
                tiff_path=tiff_path,
                png_path=png_path,
                roi_vertices=roi_vertices,
                output_prefix=output_prefix,
                b_save_pointcloud=b_save_pointcloud
            )
        except Exception as e:
            print(f"处理第 {idx} 帧时出错: {str(e)}")
            continue


def main():
    roi_vertices = np.array([
        [0, 0],  # 左上角
        [0, 5119],  # 左下角
        [5119, 5119],  # 右下角
        [5119, 0]   # 右上角
    ])
    
    # 指定数据文件夹路径
    data_folder = "./golden_wire/" 
    output_folder = Path(data_folder) / f"results"
    b_save_pointcloud = True
    start_idx = 4
    end_idx = 5
    
    process_folder(data_folder, output_folder, roi_vertices, start_idx=start_idx, end_idx=end_idx, b_save_pointcloud=b_save_pointcloud)
    

if __name__ == "__main__":
    main()


