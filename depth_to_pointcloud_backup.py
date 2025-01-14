import numpy as np
from PIL import Image
import configparser
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.path import Path
import os
from pathlib import Path
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
import cv2
from scipy import optimize

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
        return np.array(Image.open(tiff_path))
    
    def convert_to_pointcloud(self, depth_image):
        """将深度图转换为点云"""
        height, width = depth_image.shape
        points_3d = []
        points_2d = []  # 存储对应的2D坐标
        # 添加进度条
        for v in tqdm(range(height), desc="Converting to pointcloud"):
            for u in range(width):
                # 检查点是否在四边形内
                if not self.is_point_in_polygon((u, v)):
                    continue
                g = depth_image[v, u]
                # 跳过深度值为0的点
                if g == 0:
                    continue
                    
                # 根据公式计算3D坐标
                x = u * self.dx + self.ox
                y = v * self.dy + self.oy
                z = g * self.dz + self.oz
                
                points_3d.append([x, y, z])
                points_2d.append([u, v])  # 保存对应的2D坐标
                
        return np.array(points_3d), np.array(points_2d)
    
    def save_pointcloud(self, points_3d, points_2d, output_path_3d, output_path_2d):
        """将点云数据和对应的2D坐标保存为txt文件"""
        np.savetxt(output_path_3d, points_3d, fmt='%.6f', delimiter=' ')
        np.savetxt(output_path_2d, points_2d, fmt='%d', delimiter=' ')
        
    def visualize_region(self, depth_image):
        """可视化四边形区域"""
        plt.figure(figsize=(10, 8))
        plt.imshow(depth_image, cmap='jet')
        
        # 绘制四边形区域
        vertices = np.vstack((self.vertices, self.vertices[0]))  # 闭合多边形
        plt.plot(vertices[:, 0], vertices[:, 1], 'r-', linewidth=2, label='ROI')
        
        plt.colorbar()
        plt.title('Depth Map with ROI')
        plt.legend()
        plt.show()
        
    def crop_roi_from_image(self, image_path, output_prefix):
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
        cropped_path = f'{output_prefix}_cropped_roi.png'
        
        # 保存带顶点标注的图像
        plt.savefig(vertices_path)
        plt.close()
        
        # 保存原始裁剪图像
        cropped_img = Image.fromarray(cropped_image)
        cropped_img.save(cropped_path)
        
        return cropped_image, (min_x, min_y, max_x, max_y)

    def find_hemisphere_peaks(self, points_3d, points_2d, z_threshold=0.05, 
                            cluster_radius=20, min_points=3, sample_size=2000):
        """
        找出半球形突起的顶点坐标
        
        参数:
        points_3d: np.array, 形状为(N, 3)的点云数据
        points_2d: np.array, 形状为(N, 2)的对应2D像素坐标
        depth_image: np.array, 原始深度图像
        z_threshold: float, 与拟合平面的距离阈值（单位：米）
        cluster_radius: int, 半球顶点区域的半径（像素单位）
        min_points: int, 一个有效聚类所需的最小点数
        sample_size: int, 用于拟合平面的采样点数
        """
        # 转换为numpy数组
        points_3d = np.array(points_3d)
        points_2d = np.array(points_2d)
        
        # 1. 随机采样点用于平面拟合

        # 1.1 创建2D点到3D点的映射
        point_map = {(x, y): point_3d for (x, y), point_3d in zip(points_2d, points_3d)}
        
        # 1.2 计算ROI中心点
        center = np.mean(self.vertices, axis=0)
        
        # 1.3 获取四个内缩点及其周围的点
        sample_points = []
        window_size = 9  # 9x9的窗口
        half_window = window_size // 2
        # 打印四边形顶点坐标
        print("四边形顶点坐标:")
        print(self.vertices)
        for vertex in self.vertices:
            # 计算从顶点到中心的方向向量
            direction = center - vertex
            direction = direction / np.linalg.norm(direction)  # 归一化方向向量
            
            # 计算内缩11个像素的点
            inset_point = vertex + 11 * direction  # 向内移动11个像素
            inset_point = np.round(inset_point).astype(int)  # 四舍五入到整数坐标
            print(f"内缩点坐标: {inset_point}")
            # 在内缩点周围取9x9的窗口
            for i in range(-half_window, half_window + 1):
                for j in range(-half_window, half_window + 1):
                    sample_x = inset_point[0] + i
                    sample_y = inset_point[1] + j
                    sample_point = (int(sample_x), int(sample_y))
                    
                    # 如果该点存在对应的3D坐标，则添加到采样点列表
                    if sample_point in point_map:
                        sample_points.append(point_map[sample_point])
        
        # 1.4 转换为numpy数组
        sample_points = np.array(sample_points)
        
        # 打印采样点数量
        print(f"用于拟合平面的点数量: {len(sample_points)}")
        
        if len(sample_points) < 3:
            print("警告：没有足够的点来拟合平面")
            return []
        
        
        # 2. 拟合平面
        # 首先使用最小二乘法获得初始解
        A = np.column_stack((sample_points[:, 0], sample_points[:, 1], np.ones_like(sample_points[:, 0])))
        b = sample_points[:, 2]
        # 求解平面参数 [a, b, c]
        params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        
        # 获取初始平面参数
        initial_normal = np.array([-params[0], -params[1], 1])
        initial_normal = initial_normal / np.linalg.norm(initial_normal)
        initial_d = -params[2]
        
        # 将最小二乘的结果作为优化的初始值
        initial_guess = [initial_normal[0], initial_normal[1], initial_normal[2], initial_d]
        
        # 使用optimize.minimize进行优化
        def plane_error(params, points):
            a, b, c, d = params
            normal = np.array([a, b, c])
            normal = normal / np.linalg.norm(normal)
            distances = np.abs(np.dot(points, normal) + d)
            return np.mean(distances ** 2)
        
        # 优化求解
        result = optimize.minimize(
            plane_error, 
            initial_guess, 
            args=(sample_points,),
            method='Nelder-Mead'
        )
        
        # 获取最终的平面参数
        a, b, c, d = result.x
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)
        d = d / np.linalg.norm(normal)
        
        # 3. 计算所有点到平面的距离
        distances = np.abs(np.dot(points_3d, normal) + d) / np.linalg.norm(normal)
        
        # 4. 找出距离大于阈值的点
        peak_candidates_mask = distances > z_threshold
        peak_candidates_3d = points_3d[peak_candidates_mask]
        peak_candidates_2d = points_2d[peak_candidates_mask]
        
        # 打印候选点的数量
        print(f"候选点数量: {len(peak_candidates_3d)}")
        # 将候选点保存为txt文件
        peak_candidates_save_path = 'peak_candidates.txt'
        np.savetxt(peak_candidates_save_path, peak_candidates_3d, fmt='%.6f', delimiter=' ',
                  header='x y z', comments='')
        print(f"已将{len(peak_candidates_3d)}个候选点保存至 {peak_candidates_save_path}")
        
        if len(peak_candidates_3d) == 0:
            print("未找到任何候选点")
            return []
        
        # 5. 对候选点进行DBSCAN聚类
        clustering = DBSCAN(
            eps=cluster_radius,
            min_samples=min_points
        ).fit(peak_candidates_2d)
        labels = clustering.labels_
        # 打印聚类的类别数(不包括噪声点-1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"聚类数量: {n_clusters}")
        
        # 6. 处理每个聚类，找出局部最高点
        peaks = []
        # 创建4704*4704的纯黑三通道图像
        cluster_visualization = np.zeros((4704, 4704, 3), dtype=np.uint8)
        
        for label in set(labels):
            if label == -1:  # 跳过噪声点
                continue
                
            # 获取当前聚类的所有点
            cluster_mask = labels == label
            cluster_points_3d = peak_candidates_3d[cluster_mask]
            cluster_points_2d = peak_candidates_2d[cluster_mask]
            
            # 为当前聚类随机生成颜色 (避免太暗的颜色)
            color = np.random.randint(50, 255, size=3, dtype=np.uint8)
            
            # 在可视化图像上标记当前聚类的点
            for x, y in cluster_points_2d.astype(int):
                if 0 <= x < 4704 and 0 <= y < 4704:  # 确保坐标在图像范围内
                    cluster_visualization[y, x] = color
            
            # 找出聚类中Z值最大的点作为顶点
            max_z_idx = np.argmax(cluster_points_3d[:, 2])  # 使用Z坐标（第3列）
            peak_3d = cluster_points_3d[max_z_idx]
            peak_2d = cluster_points_2d[max_z_idx]
            
            # 计算该点到平面的距离作为高度
            distance = np.abs(np.dot(peak_3d, normal) + d) / np.linalg.norm(normal)
            
            peaks.append({
                '2d_coord': tuple(peak_2d.astype(int)),
                '3d_coord': peak_3d,
                'height': distance,
                'prominence': distance,
                'cluster_size': np.sum(cluster_mask)
            })
        
        # 保存聚类可视化结果
        cv2.imwrite('cluster_visualization.png', cluster_visualization)
        
        # 7. 按到平面距离排序
        peaks.sort(key=lambda x: x['height'], reverse=True)
        
        return peaks

    def calculate_hemisphere_heights(self, peaks, points_3d, points_2d, window_size=7, r_pixel=14):
        """计算半球顶点相对于平面的高度
        
        参数:
        peaks: 半球顶点列表
        points_3d: 所有3D点云数据
        points_2d: 所有对应的2D坐标
        window_size: 计算中值的窗口大小（默认7）
        r_pixel: 拟合平面的圆半径（默认14）
        
        返回:
        list: 包含每个顶点高度信息的字典列表
        """
        results = []
        half_window = window_size // 2
        
        # 创建2D点到3D点的映射
        point_map = {(x, y): point_3d for (x, y), point_3d in zip(points_2d, points_3d)}
        
        for peak in peaks:
            peak_x, peak_y = peak['2d_coord']
            
            # 1. 获取窗口内的点
            window_points = []
            for i in range(-half_window, half_window + 1):
                for j in range(-half_window, half_window + 1):
                    x, y = peak_x + i, peak_y + j
                    if (x, y) in point_map:
                        window_points.append(point_map[(x, y)])
            
            if not window_points:
                continue
            
            window_points = np.array(window_points)
            # 计算中值点
            median_z = np.median(window_points[:, 2])
            median_point = window_points[np.argmin(np.abs(window_points[:, 2] - median_z))]
            
            # 2. 获取圆周上的点
            circle_points = []
            for theta in np.linspace(0, 2*np.pi, 36):  # 每10度采样一个点
                x = int(peak_x + r_pixel * np.cos(theta))
                y = int(peak_y + r_pixel * np.sin(theta))
                if (x, y) in point_map:
                    circle_points.append(point_map[(x, y)])
            
            if len(circle_points) < 3:  # 需要至少3个点来拟合平面
                continue
            
            circle_points = np.array(circle_points)
            
            # 拟合平面 ax + by + cz + d = 0
            center = np.mean(circle_points, axis=0)
            # 使用SVD进行平面拟合
            u, s, vh = np.linalg.svd(circle_points - center)
            normal = vh[2]  # 平面法向量
            d = -np.dot(normal, center)
            
            # 3. 计算中值点到平面的距离
            distance = np.abs(np.dot(normal, median_point) + d) / np.linalg.norm(normal)
            
            results.append({
                'peak_coord': peak['2d_coord'],
                'median_point': median_point,
                'height': distance,
                'plane_normal': normal,
                'plane_d': d
            })
            
        return results

def process_single_frame(ini_path, tiff_path, png_path, roi_vertices, output_prefix='output'):
    """
    处理单帧数据的函数
    
    参数:
    ini_path: 配置文件路径 (例如: '1.ini')
    tiff_path: 深度图路径 (例如: '1.tiff')
    png_path: RGB图像路径 (例如: '1.png')
    roi_vertices: ROI顶点坐标 numpy数组
    output_prefix: 输出文件前缀 (默认: 'output')
    
    返回:
    dict: 包含处理结果的字典
    """

    # 创建转换器实例
    converter = DepthToPointCloud(vertices=roi_vertices)


    # 读取配置文件
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
    points_3d, points_2d = converter.convert_to_pointcloud(depth_image)
    
    # 保存结果
    pointcloud_3d_path = f'{output_prefix}_pointcloud_3d.txt'
    pointcloud_2d_path = f'{output_prefix}_pointcloud_2d.txt'
    converter.save_pointcloud(points_3d, points_2d, pointcloud_3d_path, pointcloud_2d_path)
    
    # 检测半球顶点
    peaks = converter.find_hemisphere_peaks(points_3d, points_2d, z_threshold=0.08, cluster_radius=5, min_points=2, sample_size=20000)
    
    # 在裁剪后的图像上标记半球顶点位置
    plt.figure(figsize=(10, 8))
    plt.imshow(cropped_image)
    print("检测到的半球顶点坐标:")
    print(f"检测到的半球顶点数量: {len(peaks)}")
    
    # 将peaks坐标转换为裁剪图像坐标系
    x_offset, y_offset = bbox[0], bbox[1]
    for peak in peaks:
        peak_x = peak['2d_coord'][0] - x_offset  # 使用字典中的坐标
        peak_y = peak['2d_coord'][1] - y_offset
        circle = plt.Circle((peak_x, peak_y), radius=1, color='red', fill=False)
        plt.gca().add_patch(circle)

    plt.title('peaks visualization')
    peaks_visualization_path = f'{output_prefix}_peaks_visualization.png'
    plt.savefig(peaks_visualization_path)
    plt.close()
    
    # 计算半球顶点相对于平面的高度
    hemisphere_heights = converter.calculate_hemisphere_heights(peaks, points_3d, points_2d, window_size=7, r_pixel=14)
    
    # 在裁剪后的图像上标注半球高度
    plt.figure(figsize=(10, 8))
    plt.imshow(cropped_image)
    
    # 将peaks坐标转换为裁剪图像坐标系并标注高度值
    for peak, height_dict in zip(peaks, hemisphere_heights):
        peak_x = peak['2d_coord'][0] - x_offset
        peak_y = peak['2d_coord'][1] - y_offset
        # 在图像上添加文本标注（将米转换为微米，并去掉单位）
        height_um = height_dict["height"] * 1000  # 将米转换为微米
        plt.text(peak_x, peak_y, f'{height_um:.0f}', 
                color='red', 
                fontsize=8,
                horizontalalignment='center',
                verticalalignment='bottom')

    plt.title('Hemisphere Heights Visualization')
    heights_visualization_path = f'{output_prefix}_heights_visualization.png'
    plt.savefig(heights_visualization_path)
    plt.close()
    
    # 更新返回结果
    results = {
        'points_3d': points_3d,
        'points_2d': points_2d,
        'cropped_image': cropped_image,
        'bbox': bbox,
        'num_points': len(points_3d),
        'hemisphere_peaks': peaks,
        'hemisphere_heights': hemisphere_heights,
        'output_files': {
            'pointcloud_3d': pointcloud_3d_path,
            'pointcloud_2d': pointcloud_2d_path,
            'cropped_image': f'{output_prefix}_cropped_roi.png',
            'cropped_image_with_vertices': f'{output_prefix}_cropped_roi_with_vertices.png',
            'peaks_visualization': peaks_visualization_path,
            'depth_visualization': f'{output_prefix}_depth_image.png'
        }
    }
    
    return results

def process_folder(folder_path, roi_vertices, start_idx=1, end_idx=50):
    """
    处理指定文件夹中的所有数据
    
    参数:
    folder_path: 数据文件夹路径
    roi_vertices: ROI顶点坐标
    start_idx: 起始索引（默认1）
    end_idx: 结束索引（默认50）
    """
    # 确保文件夹路径是字符串类型
    folder_path = str(Path(folder_path))
    
    # 创建输出文件夹
    output_folder = Path(folder_path) / "results"
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
            results = process_single_frame(
                ini_path=ini_path,
                tiff_path=tiff_path,
                png_path=png_path,
                roi_vertices=roi_vertices,
                output_prefix=output_prefix
            )
            
            # 打印处理结果
            print(f"ROI 边界框坐标: 左上({results['bbox'][0]}, {results['bbox'][1]}) "
                  f"右下({results['bbox'][2]}, {results['bbox'][3]})")
            print(f"裁剪图像尺寸: {results['cropped_image'].shape}")
            print(f"生成点数: {results['num_points']}")
            print("输出文件:")
            for key, path in results['output_files'].items():
                print(f"- {key}: {path}")
                
        except Exception as e:
            print(f"处理第 {idx} 帧时出错: {str(e)}")
            continue
            
    print("\n批处理完成！")
    print(f"结果保存在: {output_folder}")

def main():
    """
    主函数示例
    """
    # 定义四边形顶点坐标
    roi_vertices = np.array([
        [2119, 2202],  # 左上角
        [2149, 3065],  # 左下角
        [2927, 3040],  # 右下角
        [2895, 2159]   # 右上角
    ])
    
    # 指定数据文件夹路径
    data_folder = "./dataset/"  # 替换为你的实际文件夹路径
    
    # 处理文件夹中的所有数据
    try:
        process_folder(
            folder_path=data_folder,
            roi_vertices=roi_vertices,
            start_idx=1,
            end_idx=1
        )
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")

if __name__ == "__main__":
    main() 