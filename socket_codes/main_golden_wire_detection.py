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
from generate_seg_mask import calc_mask_and_medline

def create_pointcloud_converter(roi_vertices, ini_path, tiff_path, output_prefix):
    converter = DepthToPointCloud(vertices=roi_vertices)
    converter.read_config(ini_path)
    depth_image = converter.read_depth_image(tiff_path)
    # 深度图可视化处理
    depth_normalized = ((depth_image - depth_image.min()) * 255.0 / (depth_image.max() - depth_image.min())).astype(np.uint8)
    cv2.imwrite(f'{output_prefix}_depth_image_gray.png', depth_normalized)
    depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    cv2.imwrite(f'{output_prefix}_depth_image_color.png', depth_color)
    
    return converter, depth_image
    
    
def calc_depth_and_pointcloud(converter, output_prefix, b_save_pointcloud):
    """
    处理单帧的深度图和点云生成流程
    返回:
        depth_image: 原始深度图数据
        cropped_image: 裁剪后的RGB图像
        points_3d: 3D点云坐标
        points_2d: 2D投影坐标
    """    
    # 生成点云
    points_3d, points_2d = converter.convert_to_pointcloud()
    
    # 点云保存逻辑
    if b_save_pointcloud:
        pointcloud_3d_path = f'{output_prefix}_pointcloud_3d.txt'
        pointcloud_2d_path = f'{output_prefix}_pointcloud_2d.txt'
        converter.save_pointcloud(points_3d, points_2d, pointcloud_3d_path, pointcloud_2d_path)
        print(f"点云已保存至 {pointcloud_3d_path} 和 {pointcloud_2d_path}")

    return points_3d, points_2d

def convert_2d_medlines_to_3d(converter, golden_wire_medlines_2d, output_prefix, b_save_pointcloud):
    """
    将2D金线中心线坐标转换为对应的3D坐标
    
    参数:
        converter: 点云转换器实例，包含坐标转换参数和深度图数据
        golden_wire_medlines_2d: list[list] 二维列表，包含各金线的2D中心线坐标序列
        output_prefix: str 输出文件前缀路径
        b_save_pointcloud: bool 是否保存3D点云数据的标志
        
    返回:
        list[list]: 三维列表，包含各金线的3D中心线坐标序列，每个元素为numpy数组[x,y,z]
    """ 
    golden_wire_medlines_3d = []
    
    for wire_2d in golden_wire_medlines_2d:
        wire_3d = []
        for pt_2d in wire_2d:
            # 直接使用转换器进行坐标转换
            point_3d = converter.convert_single_point(
                point_2d=pt_2d
            )
            if point_3d is not None:
                wire_3d.append(point_3d)
        
        # 保留有效线段（至少两个点）
        if len(wire_3d) >= 2:
            golden_wire_medlines_3d.append(wire_3d)
    
    # 保存3D中心线
    if (b_save_pointcloud):
        with open(f'{output_prefix}_golden_wire_medlines_3d.txt', 'w') as f:
            for wire in golden_wire_medlines_3d:
                for pt in wire:
                    f.write(f"{pt[0]} {pt[1]} {pt[2]}\n")
    return golden_wire_medlines_3d

def draw_polygon_annotations(mask_txt_path, png_path, depth_image, width, height, output_prefix):
    """
    在RGB和深度图上绘制多边形标注
    
    参数:
        mask_txt_path: 多边形标注文件路径
        png_path: 原始RGB图像路径
        depth_image: 深度图数据
        width: 图像宽度
        height: 图像高度
        output_prefix: 输出文件前缀
    """
    # 初始化图像数据
    img_rgb = cv2.imread(png_path) if os.path.exists(png_path) else None
    img_depth = None
    if depth_image is not None:
        depth_normalized = ((depth_image - depth_image.min()) * 255.0 / 
                           (depth_image.max() - depth_image.min())).astype(np.uint8)
        img_depth = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    # 读取并处理多边形数据
    with open(mask_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            coords = line.split()
            xy_vals = coords[1:]
            if len(xy_vals) < 4:
                continue

            # 转换坐标
            polygon_points = []
            for i in range(0, len(xy_vals), 2):
                x = int(float(xy_vals[i]) * width)
                y = int(float(xy_vals[i+1]) * height)
                polygon_points.append([x, y])
            
            polygon_points = np.array(polygon_points, dtype=np.int32)

            # 绘制到RGB图像
            if img_rgb is not None:
                cv2.polylines(img_rgb, [polygon_points], True, (0,255,0), 8)
            
            # 绘制到深度图
            if img_depth is not None:
                cv2.polylines(img_depth, [polygon_points], True, (0,255,0), 8)

    # 保存结果
    if img_rgb is not None:
        cv2.imwrite(f'{output_prefix}_polygon_rgb.png', img_rgb)
    if img_depth is not None:
        cv2.imwrite(f'{output_prefix}_polygon_depth.png', img_depth)

def process_single_frame(ini_path, tiff_path, png_path, roi_vertices, mask_txt_path, output_prefix='output', b_save_pointcloud=False, b_calc_total_pointcloud=False):
    # 1. 创建深度图到点云转换器
    converter, depth_image = create_pointcloud_converter(roi_vertices, ini_path, tiff_path, output_prefix)
    
    # 2. 生成点云
    if b_calc_total_pointcloud:
        points_3d, points_2d = calc_depth_and_pointcloud(converter, output_prefix, b_save_pointcloud)
    
    # 2. 生成mask和中心线
    width = 5120
    height = 5120
    mask, medlines_mask, golden_wire_medlines_2d = calc_mask_and_medline(mask_txt_path, width, height, detect_method=3)
    # 保存mask和中心线mask
    cv2.imwrite(f'{output_prefix}_mask.jpg', mask)
    cv2.imwrite(f'{output_prefix}_medline_mask.jpg', medlines_mask)
        
    # 3. 将mask和中心线转换为3D坐标
    golden_wire_medlines_3d = convert_2d_medlines_to_3d(
        converter=converter,
        golden_wire_medlines_2d=golden_wire_medlines_2d,
        output_prefix=output_prefix,
        b_save_pointcloud=b_save_pointcloud
    )
    
    if 0 and b_save_pointcloud:
        draw_polygon_annotations(
            mask_txt_path=mask_txt_path,
            png_path=png_path,
            depth_image=depth_image,
            width=width,
            height=height,
            output_prefix=output_prefix
        )
    
    # 将结果整合到返回字典
    results = {
        'depth_image': depth_image,
        'mask': mask,
        'medlines_mask': medlines_mask,
        'golden_wire_medlines_2d': golden_wire_medlines_2d,
        'golden_wire_medlines_3d': golden_wire_medlines_3d
    }
    return results

def process_folder(folder_path, output_folder, roi_vertices, start_idx=1, end_idx=50, b_save_pointcloud=False, b_calc_total_pointcloud=False):
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
        mask_txt_path = str(Path(folder_path) / f"polygon{idx}.txt")
        
        # 检查文件是否存在
        if not all(Path(f).exists() for f in [ini_path, tiff_path, png_path, mask_txt_path]):
            print(f"警告: 第 {idx} 帧的某些文件不存在，跳过处理")
            continue
            
        try:
            # 设置输出前缀（包含完整路径）
            output_prefix = str(output_folder / f"frame_{idx}")
            
            # 处理单帧数据
            results = process_single_frame(
                ini_path=ini_path,
                tiff_path=tiff_path,
                png_path=png_path,
                roi_vertices=roi_vertices,
                mask_txt_path=mask_txt_path,
                output_prefix=output_prefix,
                b_save_pointcloud=b_save_pointcloud,
                b_calc_total_pointcloud=b_calc_total_pointcloud
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
    b_calc_total_pointcloud = False
    start_idx = 1
    end_idx = 1
    
    process_folder(data_folder, output_folder, roi_vertices, start_idx=start_idx, end_idx=end_idx, b_save_pointcloud=b_save_pointcloud, b_calc_total_pointcloud=b_calc_total_pointcloud)
    

if __name__ == "__main__":
    main()


