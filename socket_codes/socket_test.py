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

def fit_plane(points):
    """
    使用3D点集拟合平面，返回法向量和距离参数
    
    参数:
    points: np.array, 形状为(N, 3)的点云数据
    
    返回:
    normal: np.array, 归一化的平面法向量
    d: float, 平面方程的距离参数
    """
    # 1. 使用最小二乘法获得初始解
    A = np.column_stack((points[:, 0], points[:, 1], np.ones_like(points[:, 0])))
    b = points[:, 2]
    # 求解平面参数 [a, b, c]
    params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    # 获取初始平面参数
    initial_normal = np.array([-params[0], -params[1], 1])
    initial_normal = initial_normal / np.linalg.norm(initial_normal)
    initial_d = -params[2]
    
    # 将最小二乘的结果作为优化的初始值
    initial_guess = [initial_normal[0], initial_normal[1], initial_normal[2], initial_d]
    
    # 2. 使用optimize.minimize进行优化
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
        args=(points,),
        method='Nelder-Mead'
    )
    
    # 3. 获取最终的平面参数
    a, b, c, d = result.x
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    d = d / np.linalg.norm(normal)
    
    return normal, d

def find_hemisphere_peaks(vertices, points_3d, points_2d, bbox, output_prefix, 
                        z_threshold=0.05, cluster_radius=20, corner_window_size=9,
                        cell_size=(5, 5), peak_ratio=0.6):
    """
    找出半球形突起的顶点坐标
    
    参数:
    points_3d: np.array, 形状为(N, 3)的点云数据
    points_2d: np.array, 形状为(N, 2)的对应2D像素坐标
    bbox: tuple, (x0, y0, width, height)
    output_prefix: str, 输出文件前缀
    z_threshold: float, 最小高度阈值
    cluster_radius: int, 聚类半径
    corner_window_size: int, 角点窗口大小
    cell_size: tuple, 子区域划分大小 (行数, 列数)
    peak_ratio: float, 峰值比例系数
    """
    # 转换为numpy数组
    points_3d = np.array(points_3d)
    points_2d = np.array(points_2d)
    
    # 1. 随机采样点用于平面拟合

    # 1.1 创建2D点到3D点的映射
    point_map = {(x, y): point_3d for (x, y), point_3d in zip(points_2d, points_3d)}
    
    # 1.2 计算ROI中心点
    center = np.mean(vertices, axis=0)
    
    # 1.3 获取四个内缩点及其周围的点
    sample_points = []
    for vertex in vertices:
        # 计算从顶点到中心的方向向量
        direction = center - vertex
        direction = direction / np.linalg.norm(direction)  # 归一化方向向量
        
        # 计算内缩11个像素的点
        inset_point = vertex + (corner_window_size + 2) * direction  # 向内移动11个像素
        inset_point = np.round(inset_point).astype(int)  # 四舍五入到整数坐标

        # 在内缩点周围取9x9的窗口
        for i in range(-corner_window_size // 2, corner_window_size // 2 + 1):
            for j in range(-corner_window_size // 2, corner_window_size // 2 + 1):
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
    normal, d = fit_plane(sample_points)
    
    # 3. 计算所有点到平面的距离
    distances = np.abs(np.dot(points_3d, normal) + d) / np.linalg.norm(normal)
    
    # 4. 找出距离大于阈值的点
    # peak_candidates_mask = distances > z_threshold
    # peak_candidates_3d = points_3d[peak_candidates_mask]
    # peak_candidates_2d = points_2d[peak_candidates_mask]
    x0, y0, width, height = bbox
    cell_width = width // cell_size[0]
    cell_height = height // cell_size[1]
    
    # 初始化存储所有候选点的列表
    all_candidates_3d = []
    all_candidates_2d = []
    
    # 遍历所有子区域
    for i in range(cell_size[0]):
        for j in range(cell_size[1]):
            # 计算当前格子的边界
            cell_x_min = x0 + i * cell_width    
            cell_x_max = x0 + (i + 1) * cell_width if i < cell_size[0]-1 else x0 + width
            cell_y_min = y0 + j * cell_height
            cell_y_max = y0 + (j + 1) * cell_height if j < cell_size[1]-1 else y0 + height
            
            # 找出当前格子内的点
            cell_mask = ((points_2d[:, 0] >= cell_x_min) & 
                        (points_2d[:, 0] < cell_x_max) & 
                        (points_2d[:, 1] >= cell_y_min) & 
                        (points_2d[:, 1] < cell_y_max))
            
            # 获取当前格子内的距离值
            cell_distances = distances[cell_mask]
            
            if len(cell_distances) > 0:
                # 计算当前格子的自适应阈值（最大距离的0.6）
                cell_threshold = np.max(cell_distances) * peak_ratio
                
                # 如果自适应阈值小于全局阈值，跳过该区域
                if cell_threshold < z_threshold:
                    continue
                
                # 找出超过阈值的点
                cell_candidates_mask = distances[cell_mask] > cell_threshold
                
                # 收集候选点
                cell_candidates_3d = points_3d[cell_mask][cell_candidates_mask]
                cell_candidates_2d = points_2d[cell_mask][cell_candidates_mask]
                
                if len(cell_candidates_3d) > 0:
                    all_candidates_3d.append(cell_candidates_3d)
                    all_candidates_2d.append(cell_candidates_2d)
    
    # 合并所有格子的候选点
    peak_candidates_3d = np.concatenate(all_candidates_3d) if all_candidates_3d else np.array([])
    peak_candidates_2d = np.concatenate(all_candidates_2d) if all_candidates_2d else np.array([])
            
    # 打印候选点的数量
    print(f"候选点数量: {len(peak_candidates_3d)}")
    
    # 将候选点保存为txt文件
    peak_candidates_save_path = f'{output_prefix}_peak_candidates.txt'
    np.savetxt(peak_candidates_save_path, peak_candidates_3d, fmt='%.6f', delimiter=' ',
                header='x y z', comments='')
    print(f"已将{len(peak_candidates_3d)}个候选点保存至 {peak_candidates_save_path}")
    
    if len(peak_candidates_3d) == 0:
        print("未找到任何候选点")
        return []
    
    # 5. 对候选点进行DBSCAN聚类
    clustering = DBSCAN(
        eps=cluster_radius,
        min_samples=3
    ).fit(peak_candidates_2d)
    labels = clustering.labels_
    # 打印聚类的类别数(不包括噪声点-1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"聚类数量: {n_clusters}")
    
    # 6. 处理每个聚类，找出局部最高点
    peaks = []
    # 创建6448*6448的纯黑三通道图像
    cluster_visualization = np.zeros((6448, 6448, 3), dtype=np.uint8)
    
    for label in set(labels):
        if label == -1:  # 跳过噪声点
            continue
            
        # 获取当前聚类的所有点
        cluster_mask = labels == label
        cluster_points_3d = peak_candidates_3d[cluster_mask]
        cluster_points_2d = peak_candidates_2d[cluster_mask]
        
        # 如果聚类点数太少,跳过该聚类
        if len(cluster_points_2d) < 100: # 300, 50
            continue
        
        # 为当前聚类随机生成颜色 (避免太暗的颜色)
        color = np.random.randint(50, 255, size=3, dtype=np.uint8)
        
        # 在可视化图像上标记当前聚类的点
        for x, y in cluster_points_2d.astype(int):
            if 0 <= x < 6448 and 0 <= y < 6448:  # 确保坐标在图像范围内
                cluster_visualization[y, x] = color
        
        # 找出聚类中Z值最大的点作为顶点
        max_z_idx = np.argmax(cluster_points_3d[:, 2])  # 使用Z坐标（第3列）
        peak_3d = cluster_points_3d[max_z_idx]
        peak_2d = cluster_points_2d[max_z_idx]
        # 计算聚类的重心作为顶点
        # peak_2d = np.mean(cluster_points_2d, axis=0)
        # # 找到最接近重心的实际点
        # distances = np.linalg.norm(cluster_points_2d - peak_2d, axis=1)
        # closest_idx = np.argmin(distances)
        # peak_2d = cluster_points_2d[closest_idx]
        # peak_3d = cluster_points_3d[closest_idx]
        
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
    cluster_visualization_path = f'{output_prefix}_cluster_visualization.png'
    cv2.imwrite(cluster_visualization_path, cluster_visualization)
    
    # # 7. 按到平面距离排序
    peaks.sort(key=lambda x: x['height'], reverse=True)

    return peaks

def calculate_hemisphere_heights(peaks, points_3d, points_2d, window_size=7, r_pixel=14):
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
        
        # 1. 获取圆周上的点
        circle_points_3d = []
        circle_points_2d = []
        for theta in np.linspace(0, 2*np.pi, 36):  # 每10度采样一个点
            x = int(peak_x + r_pixel * np.cos(theta))
            y = int(peak_y + r_pixel * np.sin(theta))
            if (x, y) in point_map:
                circle_points_3d.append(point_map[(x, y)])
                circle_points_2d.append((x, y))
        
        if len(circle_points_3d) < 3:  # 需要至少3个点来拟合平面
            continue
        
        circle_points_3d = np.array(circle_points_3d)
        
        # 拟合平面
        # # 拟合平面 ax + by + cz + d = 0
        # center = np.mean(circle_points, axis=0)
        # # 使用SVD进行平面拟合
        # u, s, vh = np.linalg.svd(circle_points - center)
        # normal = vh[2]  # 平面法向量
        # d = -np.dot(normal, center)
        normal, d = fit_plane(circle_points_3d)
        
        
        # 2. 获取窗口内的点
        window_points = []
        for i in range(-half_window, half_window + 1):
            for j in range(-half_window, half_window + 1):
                x, y = peak_x + i, peak_y + j
                if (x, y) in point_map:
                    window_points.append(point_map[(x, y)])
        
        if not window_points:
            continue
        
        window_points = np.array(window_points)
        # 计算每个点到平面的距离
        distances = np.abs(np.dot(window_points, normal) + d) / np.linalg.norm(normal)
        # 取距离的中值作为最终高度
        distance = np.median(distances)
        # 找到距离最接近中值的点作为median_point
        median_point = window_points[np.argmin(np.abs(distances - distance))]

        results.append({
            'peak_coord': peak['2d_coord'],
            'median_point': median_point,
            'height': distance,
            'plane_normal': normal,
            'plane_d': d,
            'circle_points_3d': circle_points_3d,
            'circle_points_2d': circle_points_2d
        })
        
    return results

def sort_peaks_by_reference(peaks, reference_peaks_path, distance_threshold=5):
    """
    根据参考peaks文件对当前peaks进行排序
    
    参数:
    peaks: list, 当前检测到的peaks列表
    reference_peaks_path: str, 参考peaks文件路径
    distance_threshold: float, 匹配时的距离阈值（默认5像素）
    
    返回:
    list: 排序后的peaks列表
    """
    if not os.path.exists(reference_peaks_path):
        return peaks
        
    # 读取参考peaks
    existed_peaks = []
    with open(reference_peaks_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.strip().split()
            peak = {
                '2d_coord': (int(data[0]), int(data[1])),
                '3d_coord': np.array([float(data[2]), float(data[3]), float(data[4])]),
                'height': float(data[5]),
                'prominence': float(data[6]), 
                'cluster_size': int(data[7])
            }
            existed_peaks.append(peak)
    print(f"从 {reference_peaks_path} 读取了 {len(existed_peaks)} 个peaks, 用于排序")
    
    # 根据参考peaks排序
    sorted_peaks = []
    peaks_to_sort = peaks.copy()  # 创建副本避免修改原始列表
    
    for existed_peak in existed_peaks:
        ex_x, ex_y = existed_peak['2d_coord']
        
        # 寻找最近的peak
        min_dist = float('inf')
        closest_peak = None
        closest_idx = -1
        
        for i, peak in enumerate(peaks_to_sort):
            x, y = peak['2d_coord']
            dist = np.sqrt((ex_x - x)**2 + (ex_y - y)**2)
            if dist < min_dist and dist < distance_threshold:
                min_dist = dist
                closest_peak = peak
                closest_idx = i
                break
        
        if closest_peak is not None:
            sorted_peaks.append(closest_peak)
            peaks_to_sort.pop(closest_idx)
        else:
            sorted_peaks.append(existed_peak)
            print(f"-------------------未匹配到peaks,使用已存在的peaks----------------")
    
    return sorted_peaks

def calculate_base_plane(points_2d, points_3d, peaks, radius, x_offset, y_offset, output_prefix):
    """
    该函数从点云中移除peaks周围的区域，然后对剩余点拟合平面并计算距离，
    最后绘制距离分布图并返回距离数组和对应的图对象。
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 从points_2d中移除peaks周围的点
    remaining_indices = np.ones(len(points_2d), dtype=bool)

    for peak in peaks:
        peak_x, peak_y = peak['2d_coord']
        # 找出在peak周围s*s区域内的点的索引
        mask = ((points_2d[:, 0] >= peak_x - radius/2) & 
                (points_2d[:, 0] <= peak_x + radius/2) & 
                (points_2d[:, 1] >= peak_y - radius/2) & 
                (points_2d[:, 1] <= peak_y + radius/2))
        remaining_indices[mask] = False

    # 获取剩余点的points_2d和points_3d
    filtered_points_2d = points_2d[remaining_indices]
    filtered_points_3d = points_3d[remaining_indices]

    # 对剩余的3D点拟合平面
    normal, d = fit_plane(filtered_points_3d)

    # 对filtered_points_3d进行0.25比例采样
    sample_size = int(len(filtered_points_3d) * 0.25)
    sampled_indices = np.linspace(0, len(filtered_points_3d)-1, sample_size, dtype=int)
    sampled_points_3d = filtered_points_3d[sampled_indices]
    sampled_points_2d = filtered_points_2d[sampled_indices]

    # 计算采样点到平面的距离
    distances = np.abs(np.dot(sampled_points_3d, normal) + d) / np.linalg.norm(normal)

    # 绘制色谱图并返回图对象
    plt.figure(figsize=(10, 8))
    plt.scatter(sampled_points_2d[:, 0] - x_offset, 
                sampled_points_2d[:, 1] - y_offset, 
                c=distances, 
                cmap='viridis', 
                s=5)
    plt.colorbar(label='Distance to fitted plane (m)')
    plt.title('Distance Distribution Visualization')
    distance_visualization_path = f'{output_prefix}_distance_distribution.png'
    plt.savefig(distance_visualization_path)

    # 注意，这里如果要返回fig对象，一般不应调用plt.close()，否则无法在外部再使用fig
    # 如果只想保存图像不关心外部再绘制，可以在此处选择关闭
    # plt.close(fig)

    return distances, sampled_points_3d, sampled_points_2d


def process_single_frame(ini_path, tiff_path, png_path, roi_vertices, output_prefix='output',
                        corner_window_size=9, cell_size=(5, 5), peak_ratio=0.6,
                        peak_window_size=7, around_peak_radius=14, b_use_pre_peaks=False, peaks_info_path='peaks_info.txt', b_save_pointcloud=False):
    """
    处理单帧数据的函数
    
    参数:
    ini_path: 配置文件路径 (例如: '1.ini')
    tiff_path: 深度图路径 (例如: '1.tiff')
    png_path: RGB图像路径 (例如: '1_0.png')
    roi_vertices: ROI顶点坐标 numpy数组
    output_prefix: 输出文件前缀 (默认: 'output')
    corner_window_size: int, 角点窗口大小, 用于拟合全局底平面, 以寻找半球顶点
    cell_size: tuple, 子区域划分大小 (行数, 列数)
    peak_ratio: float, 峰值比例系数
    peak_window_size: int, 计算高度时的窗口大小

    around_peak_radius: int, 拟合平面时的圆半径
    save_pointcloud: bool, 是否保存点云数据（默认False）
    
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
    points_3d, points_2d = converter.convert_to_pointcloud(depth_image, bbox)
    
    # 根据开关决定是否保存点云
    if b_save_pointcloud:
        pointcloud_3d_path = f'{output_prefix}_pointcloud_3d.txt'
        pointcloud_2d_path = f'{output_prefix}_pointcloud_2d.txt'
        converter.save_pointcloud(points_3d, points_2d, pointcloud_3d_path, pointcloud_2d_path)
        print(f"已将点云保存至 {pointcloud_3d_path} 和 {pointcloud_2d_path}")
    
    # 检测半球顶点
    # 如果peaks_info文件存在, 且b_use_pre_peaks为True, 则直接读取
    if os.path.exists(peaks_info_path) and b_use_pre_peaks:
        peaks = []
        with open(peaks_info_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = line.strip().split()
                peak = {
                    '2d_coord': (int(data[0]), int(data[1])),
                    '3d_coord': np.array([float(data[2]), float(data[3]), float(data[4])]),
                    'height': float(data[5]),
                    'prominence': float(data[6]), 
                    'cluster_size': int(data[7])
                }
                peaks.append(peak)
        print(f"从 {peaks_info_path} 读取了 {len(peaks)} 个peaks, 用于计算高度")
    else:
        # 调用find_hemisphere_peaks生成peaks
        peaks = find_hemisphere_peaks(
            vertices=roi_vertices,
            points_3d=points_3d,
            points_2d=points_2d,
            bbox=bbox,
            output_prefix=output_prefix,
            z_threshold=0.06,
            cluster_radius=3,
            corner_window_size=corner_window_size,
            cell_size=cell_size,
            peak_ratio=peak_ratio
        )
        if b_use_pre_peaks:
             # 如果使用预先的peaks, 首帧保存peaks信息到文件
            with open(peaks_info_path, 'w', encoding='utf-8') as f:
                for peak in peaks:
                    f.write(f"{peak['2d_coord'][0]} {peak['2d_coord'][1]} ")
                    f.write(f"{peak['3d_coord'][0]} {peak['3d_coord'][1]} {peak['3d_coord'][2]} ")
                    f.write(f"{peak['height']} ")
                    f.write(f"{peak['prominence']} ")
                    f.write(f"{peak['cluster_size']}\n")
            print(f"已将peaks信息保存至 {peaks_info_path}")
        else:
            # 如果不使用预先的peaks,则需要按照某一个统一的reference_peaks进行排序
            peaks = sort_peaks_by_reference(peaks, peaks_info_path)
            print(f"使用读取的peaks对检测到的peaks排序,共匹配到 {len(peaks)} 个peaks")
        
    # 在裁剪后的图像上标记半球顶点位置
    plt.figure(figsize=(10, 8))
    plt.imshow(cropped_image)
    print(f"检测到的半球顶点数量: {len(peaks)}")
    
    # 将peaks坐标转换为裁剪图像坐标系
    x_offset, y_offset = bbox[0], bbox[1]
    for i, peak in enumerate(peaks):
        peak_x = peak['2d_coord'][0] - x_offset  # 使用字典中的坐标
        peak_y = peak['2d_coord'][1] - y_offset
        circle = plt.Circle((peak_x, peak_y), radius=1, color='red', fill=False)
        plt.gca().add_patch(circle)
        plt.text(peak_x + 5, peak_y + 5, str(i+1), color='red', fontsize=8)

    # 计算半球顶点相对于平面的高度
    hemisphere_heights = calculate_hemisphere_heights(
        peaks, 
        points_3d, 
        points_2d, 
        window_size=peak_window_size, 
        r_pixel=around_peak_radius
    )
    # 在裁剪后的图像上标记圆周点
    for height_dict in hemisphere_heights:
        # 获取2D圆周点
        circle_points_2d = height_dict['circle_points_2d']
        for point_2d in circle_points_2d:
            circle = plt.Circle((point_2d[0] - x_offset, point_2d[1] - y_offset), radius=1, color='green', fill=False)
            plt.gca().add_patch(circle)

    
    plt.title('peaks visualization')
    peaks_visualization_path = f'{output_prefix}_peaks_visualization.png'
    plt.savefig(peaks_visualization_path)
    plt.close()
    
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
    
    # 拟合平面，并求距离
    # 从hemisphere_heights中提取所有median_points
    median_points = np.array([h['median_point'] for h in hemisphere_heights])
    if len(median_points) >= 3:  # 需要至少3个点来拟合平面
        normal, d = fit_plane(median_points)        
        # 计算每个median_point到拟合平面的距离
        for i, height_dict in enumerate(hemisphere_heights):
            median_point = height_dict['median_point']
            
            # 计算点到平面的距离
            distance_to_median_point_plane = np.abs(np.dot(median_point, normal) + d) / np.linalg.norm(normal)
            # 将距离添加到height_dict中
            hemisphere_heights[i]['distance_to_median_point_plane'] = distance_to_median_point_plane
            
        print("已计算所有median_points到拟合平面的距离")
    else:
        print("median_points数量不足,无法拟合平面")
    
    # 计算基板误差
    distance_to_base_plane, base_plane_points_3d, base_plane_points_2d = calculate_base_plane(
        points_2d=points_2d,
        points_3d=points_3d,
        peaks=peaks,
        radius=around_peak_radius * 1.8,
        x_offset=x_offset,
        y_offset=y_offset,
        output_prefix=output_prefix
    )
    
    if b_save_pointcloud:
        converter.save_pointcloud(
            points_3d=base_plane_points_3d,
            points_2d=base_plane_points_2d,
            output_path_3d=f'{output_prefix}_base_plane_points_3d.txt',
            output_path_2d=f'{output_prefix}_base_plane_points_2d.txt'
        )
        print(f"已将基板点云保存至 {output_prefix}_base_plane_points_3d.txt 和 {output_prefix}_base_plane_points_2d.txt")
    
    # 计算基板误差的最大值
    max_distance_to_base_plane = np.max(distance_to_base_plane)
    
    
    # 更新返回结果
    results = {
        'hemisphere_peaks': peaks,
        'hemisphere_heights': hemisphere_heights,
        'max_distance_to_base_plane': max_distance_to_base_plane,
    }
    
    return results

def process_folder(folder_path, output_folder, roi_vertices, start_idx=1, end_idx=50, corner_window_size=9, cell_size=(5, 5), peak_ratio=0.6, peak_window_size=7, around_peak_radius=14, b_use_pre_peaks=False, peaks_info_path='peaks_info.txt', b_save_pointcloud=False):
    """
    处理指定文件夹中的所有数据
    
    参数:
    folder_path: 数据文件夹路径
    roi_vertices: ROI顶点坐标
    start_idx: 起始索引（默认1）
    end_idx: 结束索引（默认50）
    corner_window_size: 角点窗口大小
    cell_size: 子区域划分大小 (行数, 列数)
    peak_ratio: 峰值比例系数
    peak_window_size: 计算高度时的窗口大小
    around_peak_radius: 拟合平面时的圆半径
    """
    # 确保文件夹路径是字符串类型
    folder_path = str(Path(folder_path))
    
    # 创建输出文件夹
    output_folder.mkdir(exist_ok=True)
    
    # 创建一个列表存储所有帧的高度信息
    all_heights = []
    all_dist_to_median_point_plane = []
    all_max_distance_to_base_plane = []
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
                output_prefix=output_prefix,
                corner_window_size=corner_window_size,
                cell_size=cell_size,
                peak_ratio=peak_ratio,
                peak_window_size=peak_window_size,
                around_peak_radius=around_peak_radius,
                b_use_pre_peaks=b_use_pre_peaks,
                peaks_info_path=peaks_info_path,
                b_save_pointcloud=b_save_pointcloud
            )
            
            # 提取当前帧的高度信息
            frame_heights = [h['height'] for h in results['hemisphere_heights']]
            all_heights.append(frame_heights)
            
            # 提取当前帧到中点平面的距离信息
            frame_dist_to_median_point_plane = [h['distance_to_median_point_plane'] for h in results['hemisphere_heights']]
            all_dist_to_median_point_plane.append(frame_dist_to_median_point_plane)
            
            # 提取当前帧到基板平面的最大距离
            frame_max_distance_to_base_plane = results['max_distance_to_base_plane']
            all_max_distance_to_base_plane.append(frame_max_distance_to_base_plane)
            
        except Exception as e:
            print(f"处理第 {idx} 帧时出错: {str(e)}")
            # 如果处理失败，添加空值保持行数一致
            # 获取第一个成功处理的帧中半球的数量作为基准
            base_count = next((len(h) for h in all_heights if len(h) > 0), 0)
            all_heights.append([np.nan] * base_count)
            all_dist_to_median_point_plane.append([np.nan] * base_count)
            continue
    
    # 将所有高度信息保存为CSV文件
    heights_path = str(output_folder / "heights_data.csv")
    np.savetxt(heights_path, all_heights, delimiter=',', fmt='%.6f')
    print(f"\n高度数据已保存至: {heights_path}")

    # 将所有到中点平面的距离信息保存为CSV文件
    dist_to_median_point_plane_path = str(output_folder / "dist_to_plane_data.csv")
    np.savetxt(dist_to_median_point_plane_path, all_dist_to_median_point_plane, delimiter=',', fmt='%.6f')
    print(f"\n到中点平面的距离数据已保存至: {dist_to_median_point_plane_path}")

    # 将所有到基板平面的最大距离信息保存为CSV文件
    max_distance_to_base_plane_path = str(output_folder / "max_distance_to_base_plane_data.csv")
    np.savetxt(max_distance_to_base_plane_path, all_max_distance_to_base_plane, delimiter=',', fmt='%.6f')
    print(f"\n到基板平面的最大距离数据已保存至: {max_distance_to_base_plane_path}")

def main():
    """
    主函数示例
    """
# -----------------6500w camera_4th------------
    # # center
    # key = "center"
    # roi_vertices = np.array([
    #     [2520, 2503],  # 左上角
    #     [2503, 3723],  # 左下角
    #     [3926, 3731],  # 右下角
    #     [3923, 2477]   # 右上角
    # ])
    # # 寻找球顶点的参数
    # corner_window_size = 9
    # cell_size = (5, 5)
    # peak_ratio = 0.68
    # # 计算半球顶点高度参数
    # peak_window_size = 7
    # around_peak_radius = 25
    
    # # left top
    # key = "left_top"
    # roi_vertices = np.array([
    #     [506, 482],  # 左上角
    #     [478, 1707],  # 左下角
    #     [1903, 1722],  # 右下角
    #     [1909, 460]   # 右上角
    # ])
    # # 寻找球顶点的参数
    # corner_window_size = 9
    # cell_size = (5, 5)
    # peak_ratio = 0.68
    # # 计算半球顶点高度参数
    # peak_window_size = 7
    # around_peak_radius = 25
    
    # # left bottom
    # key = "left_bottom"
    # roi_vertices = np.array([
    #     [239, 4826],  # 左上角
    #     [241, 6067],  # 左下角
    #     [2227, 6050],  # 右下角
    #     [2212, 4737]   # 右上角
    # ])
    # # 寻找球顶点的参数
    # corner_window_size = 9
    # cell_size = (5, 5)
    # peak_ratio = 0.68
    # # 计算半球顶点高度参数
    # peak_window_size = 7
    # around_peak_radius = 53
    
    # # right-top
    # key = "right_top"
    # roi_vertices = np.array([
    #     [4837, 405],  # 左上角
    #     [4812, 1636],  # 左下角
    #     [6244, 1642],  # 右下角
    #     [6248, 384]   # 右上角
    # ])
    # # 寻找球顶点的参数
    # corner_window_size = 9
    # cell_size = (5, 5)
    # peak_ratio = 0.68
    # # 计算半球顶点高度参数
    # peak_window_size = 7
    # around_peak_radius = 25
    
    # right-bottom
    key = "right_bottom"
    roi_vertices = np.array([
        [4262, 4870],  # 左上角
        [4253, 6112],  # 左下角
        [6238, 6116],  # 右下角
        [6245, 4819]   # 右上角
    ])
    # 寻找球顶点的参数
    corner_window_size = 9
    cell_size = (5, 5)
    peak_ratio = 0.68
    # 计算半球顶点高度参数
    peak_window_size = 7
    around_peak_radius = 53
    # ---------------------------------------------
    
    # 指定数据文件夹路径
    data_folder = "./dataset4/" 
    output_folder = Path(data_folder) / f"results_{key}"
    peaks_info_path = Path(data_folder) / f'peaks_info_{key}.txt'
    b_use_pre_peaks = True
    b_save_pointcloud = True
 
    # 处理文件夹中的所有数据
    try:
        process_folder(
            folder_path=data_folder,
            output_folder=output_folder,
            roi_vertices=roi_vertices,
            start_idx=1,
            end_idx=40,
            corner_window_size=corner_window_size,
            cell_size=cell_size,
            peak_ratio=peak_ratio,
            peak_window_size=peak_window_size,
            around_peak_radius=around_peak_radius,
            b_use_pre_peaks=b_use_pre_peaks,
            peaks_info_path=peaks_info_path,
            b_save_pointcloud=b_save_pointcloud
        )
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")

if __name__ == "__main__":
    main() 
    
    
#######################################################################
# -----------------2500w camera-----------------
    # 定义四边形顶点坐标
    # center
    # roi_vertices = np.array([
    #     [2119, 2202],  # 左上角
    #     [2149, 3065],  # 左下角
    #     [2927, 3040],  # 右下角
    #     [2895, 2159]   # 右上角
    # ])
    # # 寻找球顶点的参数
    # corner_window_size = 9
    # cell_size = (5, 5)
    # peak_ratio = 0.68
    # # 计算半球顶点高度参数
    # peak_window_size = 7
    # around_peak_radius = 14
    
    # left-top
    # roi_vertices = np.array([
    #     [101, 410],  # 左上角
    #     [142, 1132],  # 左下角
    #     [980, 1077],  # 右下角
    #     [929, 336]   # 右上角
    # ])
    # # 寻找球顶点的参数
    # corner_window_size = 8
    # cell_size = (5, 5)
    # peak_ratio = 0.68
    # # 计算半球顶点高度参数
    # peak_window_size = 7
    # around_peak_radius = 17
    
    # right-top
    # roi_vertices = np.array([
    #     [3842, 289],  # 左上角
    #     [3855, 1004],  # 左下角
    #     [4688, 988],  # 右下角
    #     [4666, 252]   # 右上角
    # ])
    # # 寻找球顶点的参数
    # corner_window_size = 8
    # cell_size = (5, 5)
    # peak_ratio = 0.68
    # # 计算半球顶点高度参数
    # peak_window_size = 7
    # around_peak_radius = 17

    # right-bottom
    # roi_vertices = np.array([
    #     [3373, 3595],  # 左上角
    #     [3261, 4316],  # 左下角
    #     [4407, 4506],  # 右下角
    #     [4526, 3749]   # 右上角
    # ])
    # # 寻找球顶点的参数
    # corner_window_size = 8
    # cell_size = (5, 5)
    # peak_ratio = 0.68
    # # 计算半球顶点高度参数
    # peak_window_size = 7
    # around_peak_radius = 27

    # left-bottom
    # roi_vertices = np.array([
    #     [256, 3897],  # 左上角
    #     [210, 4616],  # 左下角
    #     [1040, 4661],  # 右下角
    #     [1082, 3929]   # 右上角
    # ])
    # # 寻找球顶点的参数
    # corner_window_size = 8
    # cell_size = (5, 5)
    # peak_ratio = 0.68
    # # 计算半球顶点高度参数
    # peak_window_size = 7
    # around_peak_radius = 17

    # -----------------6500w camera-----------------

    # left-top
    # roi_vertices = np.array([
    #     [496, 578],  # 左上角
    #     [506, 1831],  # 左下角
    #     [1910, 1799],  # 右下角
    #     [1916, 569]   # 右上角
    # ])
    # # 寻找球顶点的参数
    # corner_window_size = 9
    # cell_size = (5, 5)
    # peak_ratio = 0.68
    # # 计算半球顶点高度参数
    # peak_window_size = 7
    # around_peak_radius = 23
    
    # left-bottom
    # roi_vertices = np.array([
    #     [375, 4821],  # 左上角
    #     [375, 6100],  # 左下角
    #     [2312, 6065],  # 右下角
    #     [2339, 4811]   # 右上角
    # ])
    # # 寻找球顶点的参数
    # corner_window_size = 9
    # cell_size = (5, 5)
    # peak_ratio = 0.68
    # # 计算半球顶点高度参数
    # peak_window_size = 7
    # around_peak_radius = 53
    
    # # center
    # roi_vertices = np.array([
    #     [2196, 2940],  # 左上角
    #     [2207, 4186],  # 左下角
    #     [3600, 4156],  # 右下角
    #     [3602, 2937]   # 右上角
    # ])
    # # 寻找球顶点的参数
    # corner_window_size = 9
    # cell_size = (5, 5)
    # peak_ratio = 0.68
    # # 计算半球顶点高度参数
    # peak_window_size = 7
    # around_peak_radius = 23
    
    # right-top
    # roi_vertices = np.array([
    #     [3958, 496],  # 左上角
    #     [3982, 1794],  # 左下角
    #     [5916, 1730],  # 右下角
    #     [5926, 467]   # 右上角
    # ])
    # # 寻找球顶点的参数
    # corner_window_size = 9
    # cell_size = (5, 5)
    # peak_ratio = 0.68
    # # 计算半球顶点高度参数
    # peak_window_size = 7
    # around_peak_radius = 50
    
    # right-bottom
    # roi_vertices = np.array([
    #     [3805, 4733],  # 左上角
    #     [3786, 6025],  # 左下角
    #     [5720, 6016],  # 右下角
    #     [5769, 4756]   # 右上角
    # ])
    # # 寻找球顶点的参数
    # corner_window_size = 9
    # cell_size = (5, 5)
    # peak_ratio = 0.68
    # # 计算半球顶点高度参数
    # peak_window_size = 7
    # around_peak_radius = 45
    
    # -----------------6500w camera_2nd------------
    # center
    # roi_vertices = np.array([
    #     [2009, 2405],  # 左上角
    #     [2053, 3654],  # 左下角
    #     [3456, 3590],  # 右下角
    #     [3427, 2366]   # 右上角
    # ])
    # # 寻找球顶点的参数
    # corner_window_size = 9
    # cell_size = (5, 5)
    # peak_ratio = 0.68
    # # 计算半球顶点高度参数
    # peak_window_size = 7
    # around_peak_radius = 23
    
    # left top
    # roi_vertices = np.array([
    #     [217, 926],  # 左上角
    #     [275, 2146],  # 左下角
    #     [1655, 2080],  # 右下角
    #     [1604, 866]   # 右上角
    # ])
    # # 寻找球顶点的参数
    # corner_window_size = 9
    # cell_size = (5, 5)
    # peak_ratio = 0.68
    # # 计算半球顶点高度参数
    # peak_window_size = 7
    # around_peak_radius = 23
    
    # left bottom
    # roi_vertices = np.array([
    #     [190, 4285],  # 左上角
    #     [181, 5552],  # 左下角
    #     [2110, 5569],  # 右下角
    #     [2154, 4311]   # 右上角
    # ])
    # # 寻找球顶点的参数
    # corner_window_size = 9
    # cell_size = (5, 5)
    # peak_ratio = 0.68
    # # 计算半球顶点高度参数
    # peak_window_size = 7
    # around_peak_radius = 50
    
    # right-top
    # roi_vertices = np.array([
    #     [3707, 855],  # 左上角
    #     [3724, 2161],  # 左下角
    #     [5665, 2103],  # 右下角
    #     [5689, 823]   # 右上角
    # ])
    # # 寻找球顶点的参数
    # corner_window_size = 9
    # cell_size = (5, 5)
    # peak_ratio = 0.68
    # # 计算半球顶点高度参数
    # peak_window_size = 7
    # around_peak_radius = 45
    
    # right-bottom
    # roi_vertices = np.array([
    #     [3991, 4082],  # 左上角
    #     [3983, 5369],  # 左下角
    #     [5906, 5377],  # 右下角
    #     [5953, 4117]   # 右上角
    # ])
    # # 寻找球顶点的参数
    # corner_window_size = 9
    # cell_size = (5, 5)
    # peak_ratio = 0.68
    # # 计算半球顶点高度参数
    # peak_window_size = 7
    # around_peak_radius = 52
    
    # -----------------6500w camera_3rd------------
    # # center
    # key = "center"
    # roi_vertices = np.array([
    #     [2262, 2614],  # 左上角
    #     [2221, 3825],  # 左下角
    #     [3638, 3868],  # 右下角
    #     [3655, 2622]   # 右上角
    # ])
    # # 寻找球顶点的参数
    # corner_window_size = 9
    # cell_size = (5, 5)
    # peak_ratio = 0.68
    # # 计算半球顶点高度参数
    # peak_window_size = 7
    # around_peak_radius = 25
    
    # # left top
    # key = "left_top"
    # roi_vertices = np.array([
    #     [546, 662],  # 左上角
    #     [474, 1903],  # 左下角
    #     [2439, 1976],  # 右下角
    #     [2471, 690]   # 右上角
    # ])
    # # 寻找球顶点的参数
    # corner_window_size = 9
    # cell_size = (5, 5)
    # peak_ratio = 0.68
    # # 计算半球顶点高度参数
    # peak_window_size = 7
    # around_peak_radius = 53
    
    # # left bottom
    # key = "left_bottom"
    # roi_vertices = np.array([
    #     [512, 4506],  # 左上角
    #     [551, 5746],  # 左下角
    #     [2510, 5724],  # 右下角
    #     [2493, 4440]   # 右上角
    # ])
    # # 寻找球顶点的参数
    # corner_window_size = 9
    # cell_size = (5, 5)
    # peak_ratio = 0.68
    # # 计算半球顶点高度参数
    # peak_window_size = 7
    # around_peak_radius = 53
    
    # # right-top
    # key = "right_top"
    # roi_vertices = np.array([
    #     [4126, 741],  # 左上角
    #     [4091, 1959],  # 左下角
    #     [5508, 1989],  # 右下角
    #     [5529, 747]   # 右上角
    # ])
    # # 寻找球顶点的参数
    # corner_window_size = 9
    # cell_size = (5, 5)
    # peak_ratio = 0.68
    # # 计算半球顶点高度参数
    # peak_window_size = 7
    # around_peak_radius = 25
    
    # right-bottom
    # key = "right_bottom"
    # roi_vertices = np.array([
    #     [4040, 4558],  # 左上角
    #     [3985, 5820],  # 左下角
    #     [5952, 5848],  # 右下角
    #     [5971, 4564]   # 右上角
    # ])
    # # 寻找球顶点的参数
    # corner_window_size = 9
    # cell_size = (5, 5)
    # peak_ratio = 0.68
    # # 计算半球顶点高度参数
    # peak_window_size = 7
    # around_peak_radius = 53

    # -----------------6500w camera_4th_small------------
    # # center
    # key = "center"
    # roi_vertices = np.array([
    #     [2599, 2844],  # 左上角
    #     [2578, 4080],  # 左下角
    #     [4012, 4078],  # 右下角
    #     [4012, 2818]   # 右上角
    # ])
    # # 寻找球顶点的参数
    # corner_window_size = 9
    # cell_size = (5, 5)
    # peak_ratio = 0.68
    # # 计算半球顶点高度参数
    # peak_window_size = 7
    # around_peak_radius = 25
    
    # left top
    # key = "left_top"
    # roi_vertices = np.array([
    #     [1298, 427],  # 左上角
    #     [1253, 1653],  # 左下角
    #     [2683, 1686],  # 右下角
    #     [2700, 430]   # 右上角
    # ])
    # # 寻找球顶点的参数
    # corner_window_size = 9
    # cell_size = (5, 5)
    # peak_ratio = 0.68
    # # 计算半球顶点高度参数
    # peak_window_size = 7
    # around_peak_radius = 25
    
    # # right-top
    key = "right_top"
    roi_vertices = np.array([
        [3940, 496],  # 左上角
        [3896, 1719],  # 左下角
        [5322, 1754],  # 右下角
        [5345, 496]   # 右上角
    ])
    # 寻找球顶点的参数
    corner_window_size = 9
    cell_size = (5, 5)
    peak_ratio = 0.68
    # 计算半球顶点高度参数
    peak_window_size = 7
    around_peak_radius = 25