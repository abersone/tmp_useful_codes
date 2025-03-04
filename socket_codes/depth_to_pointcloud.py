import numpy as np
from PIL import Image
import configparser
import matplotlib
matplotlib.use('Agg')  # 在导入pyplot之前设置
import matplotlib.pyplot as plt
from tqdm import tqdm


class DepthToPointCloud:
    def __init__(self, vertices=None):
        self.ox = 0.0
        self.oy = 0.0
        self.oz = 0.0
        self.dx = 0.0
        self.dy = 0.0
        self.dz = 0.0
        # 如果没有提供顶点坐标，使用默认值
        self.vertices = vertices if vertices is not None else np.array([
            [100, 100],  # 左上角
            [100, 300],  # 左下角
            [300, 300],  # 右下角
            [300, 100]   # 右上角
        ])
        self.peaks_num = -1
        self.depth_image = None
        
    def is_point_in_polygon(self, point):
        """判断点是否在四边形内
        使用射线法判断点是否在多边形内部
        参数:
            point: [x, y] 待判断的点坐标
        返回:
            bool: 点是否在多边形内部
        """
        x, y = point
        n = len(self.vertices)
        inside = False
        
        # 射线法判断点是否在多边形内
        j = n - 1
        for i in range(n):
            if ((self.vertices[i][1] > y) != (self.vertices[j][1] > y) and
                (x < (self.vertices[j][0] - self.vertices[i][0]) * 
                 (y - self.vertices[i][1]) / (self.vertices[j][1] - self.vertices[i][1]) + 
                 self.vertices[i][0])):
                inside = not inside
            j = i
            
        return inside
        
    def read_config(self, ini_path):
        """读取配置文件中的参数"""
        config = configparser.ConfigParser()
        config.read(ini_path)
        
        self.ox = float(config['CloudHeader']['ox'])
        self.oy = float(config['CloudHeader']['oy'])
        self.oz = float(config['CloudHeader']['oz'])
        self.dx = float(config['CloudHeader']['dx'])
        self.dy = float(config['CloudHeader']['dy'])
        self.dz = float(config['CloudHeader']['dz'])
        
    def read_depth_image(self, tiff_path):
        """读取32位深度图像"""
        self.depth_image = np.array(Image.open(tiff_path))
        return self.depth_image
    
    def convert_to_pointcloud(self, bbox=[]):
        """将深度图转换为点云"""
        # 计算ROI的最小外接矩形
        if len(bbox) == 4:
            min_x = bbox[0]
            max_x = bbox[0] + bbox[2]
            min_y = bbox[1]
            max_y = bbox[1] + bbox[3]
        else:
            min_x = int(np.min(self.vertices[:, 0]))
            max_x = int(np.max(self.vertices[:, 0]))
            min_y = int(np.min(self.vertices[:, 1]))
            max_y = int(np.max(self.vertices[:, 1]))

        points_3d = []
        points_2d = []  # 存储对应的2D坐标
        
        # 只在最小外接矩形范围内遍历
        for v in tqdm(range(min_y, max_y + 1), desc="Converting to pointcloud"):
            for u in range(min_x, max_x + 1):
                # 检查点是否在四边形内
                if not self.is_point_in_polygon((u, v)):
                    continue
                g = self.depth_image[v, u]
                # 跳过深度值为0的点
                if g == 0:
                    continue
                    
                # 根据公式计算3D坐标
                x = u * self.dx + self.ox
                y = v * self.dy + self.oy
                z = g * self.dz + self.oz
                y = -y
                points_3d.append([x, y, z])
                points_2d.append([u, v])
                
        return np.array(points_3d), np.array(points_2d)
    
    def save_pointcloud(self, points_3d, points_2d, output_path_3d, output_path_2d):
        """将点云数据和对应的2D坐标保存为txt文件"""
        np.savetxt(output_path_3d, points_3d, fmt='%.6f', delimiter=' ')
        np.savetxt(output_path_2d, points_2d, fmt='%d', delimiter=' ')
        
        
    def crop_roi_from_image(self, image_path, output_prefix, save_vertices_plot=False):
        """从原始图像中裁剪ROI区域并绘制顶点"""
        # 读取原始图像
        original_image = np.array(Image.open(image_path))
        
        # 计算最小外接矩形
        min_x = int(np.min(self.vertices[:, 0]))
        max_x = int(np.max(self.vertices[:, 0]))
        min_y = int(np.min(self.vertices[:, 1]))
        max_y = int(np.max(self.vertices[:, 1]))
        
        # 裁剪图像
        cropped_image = original_image[min_y:max_y, min_x:max_x]
        
        if save_vertices_plot:
            # 创建图像显示
            plt.figure(figsize=(10, 8))
            plt.imshow(cropped_image)
            
            # 调整顶点坐标到裁剪后的坐标系
            adjusted_vertices = self.vertices.copy()
            adjusted_vertices[:, 0] -= min_x
            adjusted_vertices[:, 1] -= min_y
            
            # 绘制顶点和连线
            vertices = np.vstack((adjusted_vertices, adjusted_vertices[0]))  # 闭合多边形
            plt.plot(vertices[:, 0], vertices[:, 1], 'r-', linewidth=2, label='ROI')
            
            # 绘制顶点并添加标签
            for i, (x, y) in enumerate(adjusted_vertices):
                plt.plot(x, y, 'ro', markersize=10)
                plt.text(x+10, y+10, f'P{i+1}', color='red', fontsize=12)
            
            plt.title('Cropped Image with ROI Vertices')
            plt.axis('on')
            plt.legend()
            
            # 使用output_prefix构建保存路径
            vertices_path = f'{output_prefix}_cropped_roi_with_vertices.png'
            
            # 保存带顶点标注的图像
            plt.savefig(vertices_path)
            plt.close()
        
        # 保存原始裁剪图像
        cropped_path = f'{output_prefix}_cropped_roi.png'
        cropped_img = Image.fromarray(cropped_image)
        cropped_img.save(cropped_path)
        
        return cropped_image, (min_x, min_y, max_x - min_x, max_y - min_y)

    def convert_single_point(self, point_2d):
        """
        将单个2D点转换为3D坐标
        
        参数:
             point_2d: 二维坐标 [u, v] 或 (u, v)
        
        返回:
             numpy数组: 三维坐标 [x, y, z]，如果无效则返回None
        """
        # 转换为整数坐标
        u = int(round(point_2d[0]))
        v = int(round(point_2d[1]))
        
        # 检查坐标有效性
        if not (0 <= v < self.depth_image.shape[0] and 0 <= u < self.depth_image.shape[1]):
            return None
        
        # 检查点是否在ROI多边形内
        if not self.is_point_in_polygon((u, v)):
            return None
        
        # 获取深度值
        g = self.depth_image[v, u]
        if g == 0:
            return None
        
        # 计算3D坐标
        x = u * self.dx + self.ox
        y = v * self.dy + self.oy
        z = g * self.dz + self.oz
        y = -y
        return np.array([x, y, z])

import os
def main():
    # 读取配置文件
    ini_path = "./dataset/1.ini"
    tiff_path = "./dataset/1.tiff"
    png_path = "./dataset/1_0.png"
    output_prefix = "./dataset/output/1"
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    roi_vertices = np.array([
        [0, 0],  # 左上角
        [0, 5120],  # 左下角
        [5120, 5120],  # 右下角
        [5120, 0]   # 右上角
    ])
    
    converter = DepthToPointCloud(vertices=roi_vertices)
    converter.read_config(ini_path)
    
    # 读取深度图
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
    if (1):
        points_3d, points_2d = converter.convert_to_pointcloud(bbox)
        # 保存点云
        converter.save_pointcloud(points_3d, points_2d, output_prefix + "_3d.txt", output_prefix + "_2d.txt")

if __name__ == "__main__":
    main() 