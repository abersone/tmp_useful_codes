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
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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

def plot_3d_wire_sections(wire_features, output_prefix, b_draw_direction=True, b_draw_endpoints=True, occlusion_points=None):
    """
    改进版三维可视化函数
    参数:
        wire_features: list of dict 金线特征列表，每个元素包含：
            - points: np.array 三维点集
            - head: np.array 头部端点坐标
            - tail: np.array 尾部端点坐标
        output_prefix: str 输出文件前缀
    """
    # 初始化处理
    if occlusion_points is None:
        occlusion_points = []  # 转换为空列表
    
    # 初始化图形
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 颜色配置
    cmap = plt.get_cmap('tab10')  # 使用定性色系
    wire_color = [0.2, 0.2, 0.2]   # 点云基础颜色
    head_color = [1, 0, 0]        # 头部端点颜色（红色）
    tail_color = [0, 0.5, 1]      # 尾部端点颜色（天蓝色）

    # 遍历所有金线特征
    for idx, wf in enumerate(wire_features):
        points = wf['points']
        head = wf['head']
        tail = wf['tail']
        
        # 跳过无效数据
        if len(points) < 2:
            continue

        # 计算中点位置和方向向量
        mid_point = np.median(points, axis=0) #points[len(points)//2]  # 使用点云实际中点
        direction = tail - head
        direction_normalized = direction / np.linalg.norm(direction) * 3  # 缩短为3毫米

        # 绘制点云（带透明度）
        ax.scatter(points[:,0], points[:,1], points[:,2],
                   color=cmap(idx%10),  # 10种颜色循环使用
                   s=2, 
                   alpha=0.4,
                   linewidths=0.25,      # 调小线宽
                   label=f'Segment {idx+1}')

        # 绘制头部端点（五角星标记）
        if b_draw_endpoints:
            ax.scatter(head[0], head[1], head[2],
                  color=head_color,
                  s=50,
                  marker='*',
                  edgecolor='black',
                  linewidth=0.5,
                  zorder=4)  # 确保在最上层

        # 绘制尾部端点（三角形标记）
        if b_draw_endpoints:
            ax.scatter(tail[0], tail[1], tail[2],
                  color=tail_color,
                  s=50,
                  marker='^',
                  edgecolor='black',
                  linewidth=0.5,
                  zorder=4)

        # 绘制端点连线（虚线）
        if 0:
            ax.plot([head[0], tail[0]], 
               [head[1], tail[1]], 
               [head[2], tail[2]],
               color=cmap(idx%10),
               linestyle='--',
               alpha=0.6,
               linewidth=1.5)

        # 修改后的方向向量绘制（箭头从中点出发）
        if b_draw_direction:
            ax.quiver(
                mid_point[0], mid_point[1], mid_point[2],
                direction_normalized[0], 
                direction_normalized[1],
                direction_normalized[2],
                color=cmap(idx%10),
                linewidth=1.0,        # 增加线宽
            arrow_length_ratio=0.2,  # 增大箭头比例
            length=0.05,             # 增加总长度到5毫米
            alpha=1.0,            # 最大不透明度
            zorder=5              # 确保在最顶层
            )

    # 坐标轴设置
    ax.set_xlabel('X (mm)', fontsize=11, labelpad=8)
    ax.set_ylabel('Y (mm)', fontsize=11, labelpad=8)
    ax.set_zlabel('Z (mm)', fontsize=11, labelpad=8)
    ax.xaxis.pane.fill = False  # 透明背景
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.3)   # 半透明网格

    # 创建自定义图例
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', label='Head',
               markerfacecolor=head_color, markersize=12, markeredgecolor='black'),
        Line2D([0], [0], marker='^', color='w', label='Tail',
               markerfacecolor=tail_color, markersize=12, markeredgecolor='black'),
        Patch(facecolor='gray', alpha=0.5, label='Point Cloud')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    # 修改后的遮挡点绘制逻辑
    # if occlusion_points:  # 自动处理空列表和None的情况
    #     # 统一转换为numpy数组
    #     occlusion_array = np.array(occlusion_points)
    #     ax.scatter(occlusion_array[:,0], 
    #               occlusion_array[:,1],
    #               occlusion_array[:,2],
    #               c='red', 
    #               marker='*', 
    #               s=80,
    #               edgecolor='black',
    #               linewidth=0.8,
    #               alpha=0.9,
    #               zorder=5,
    #               label='Occlusion Points')
    # 保存输出
    plt.savefig(f'{output_prefix}_wire_3d.png', 
               dpi=300, 
               bbox_inches='tight',
               transparent=True)
    plt.close()

def visualize_merged_wires(wire_groups, output_prefix, b_draw_endpoints=True, b_draw_direction=True, occlusion_points=None):
    # 准备可视化数据
    merged_features = []
    for group in wire_groups:
        if not group:
            continue
            
        # 合并点云
        merged_pts = np.concatenate([seg['points'] for seg in group], axis=0)
        
        # 计算主方向
        main_dir = compute_pca_direction(merged_pts)
        
        # 确定端点（考虑方向一致性）
        first_seg = group[0]
        last_seg = group[-1]
        head = first_seg['head'] if np.dot(main_dir, first_seg['tail'] - first_seg['head']) > 0 else first_seg['tail']
        tail = last_seg['tail'] if np.dot(main_dir, last_seg['tail'] - last_seg['head']) > 0 else last_seg['head']

        merged_features.append({
            'points': merged_pts,
            'direction': main_dir,
            'head': head,
            'tail': tail
        })

    # 生成可视化图表
    plot_3d_wire_sections(
        merged_features, 
        f"{output_prefix}", 
        b_draw_direction=b_draw_direction,  # 显示方向箭头
        b_draw_endpoints=b_draw_endpoints,   # 显示端点标记
        occlusion_points=occlusion_points
    )

def compute_pca_direction(points):
    cov_matrix = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    direction = eigenvectors[:, -1]  # 获取原始方向向量
    
    # 新增归一化处理
    norm = np.linalg.norm(direction)
    if norm > 1e-6:  # 避免除以零
        direction = direction / norm
    else:
        direction = np.zeros_like(direction)
    
    return direction

def distance_to_ray(point, ray_origin, ray_direction):
    """计算点到射线的距离（仅返回距离值）"""
    vec = point - ray_origin
    denominator = np.dot(ray_direction, ray_direction) + 1e-8
    t = np.dot(vec, ray_direction) / denominator
    
    if t <= 0:
        return np.linalg.norm(vec), t  # 返回标量距离
    else:
        projection = ray_origin + t * ray_direction
        return np.linalg.norm(point - projection), t  # 返回标量距离

def ray_distance(ray1_origin, ray1_dir, ray2_origin, ray2_dir):
    """
    改进版射线间最短距离计算（考虑射线方向约束）
    """
    w0 = ray1_origin - ray2_origin
    a = np.dot(ray1_dir, ray1_dir)
    b = np.dot(ray1_dir, ray2_dir)
    c = np.dot(ray2_dir, ray2_dir)
    d = np.dot(ray1_dir, w0)
    e = np.dot(ray2_dir, w0)     
    denominator = a*c - b*b

    # 计算理论最优参数
    s = (b*e - c*d) / denominator
    t = (a*e - b*d) / denominator
    
    # 情况1：射线平行或接近平行
    epsilon = 1e-6
    if abs(denominator) < epsilon:
        # 计算四种边界情况的最小距离
        dist1, _ = distance_to_ray(ray2_origin, ray1_origin, ray1_dir)
        dist2, _ = distance_to_ray(ray1_origin, ray2_origin, ray2_dir)
        return min(dist1, dist2, np.linalg.norm(w0)), s, t
    

    
    # 情况2：两个参数都非负
    if s >= 0 and t >= 0:
        P = ray1_origin + s*ray1_dir
        Q = ray2_origin + t*ray2_dir
        return np.linalg.norm(P - Q), s, t
    
    # 情况3：需要处理边界条件
    candidates = []
    
    # 子情况3a: s<0时，计算射线顶点1到射线2的距离
    if s < 0:
        # 等价于：计算ray1起点到ray2射线的距离
        dist, _ = distance_to_ray(ray1_origin, ray2_origin, ray2_dir)
        candidates.append(dist)
    
    # 子情况3b: t<0时，计算射线顶点2到射线1的距离
    if t < 0:
        # 等价于：计算ray2起点到ray1射线的距离
        dist, _ = distance_to_ray(ray2_origin, ray1_origin, ray1_dir)
        candidates.append(dist)
    
    # 子情况3c: s=0且t=0
    candidates.append(np.linalg.norm(w0))
    
    return min(candidates), s, t

def merge_golden_wires(golden_wire_medlines_3d, dist_threshold=0.1, output_prefix='output'):
    """
    合并属于同一条金线的线段
    参数:
        golden_wire_medlines_3d: 三维列表，每个元素是金线点集[np.array([x,y,z]), ...]
        dist_threshold: 合并阈值（单位：毫米）
    返回:
        list: 合并后的金线列表，每个元素是合并后的点集np.array
    """

    # 步骤1：预处理所有线段特征，计算主方向，确定端点
    wire_features = []
    for wire in golden_wire_medlines_3d:
        if len(wire) < 2:
            continue
        
        # 转换为numpy数组
        points = np.array(wire)
        
        # 计算主方向
        direction = compute_pca_direction(points)
 
        # 确定端点（头部和尾部）：取距离均值最近的真实点
        points_num = len(points)
        select_num = int(points_num * 0.2)
        head_points = points[:select_num]  # 取前select_num个点计算头部
        tail_points = points[-select_num:] # 取后select_num个点计算尾部

        head_center_mean = np.mean(head_points, axis=0)
        distances = np.linalg.norm(head_points - head_center_mean, axis=1)
        head_center = head_points[np.argmin(distances)]
               
        tail_center_mean = np.mean(tail_points, axis=0)
        distances = np.linalg.norm(tail_points - tail_center_mean, axis=1)
        tail_center = tail_points[np.argmin(distances)]
        
        # 确保方向向量指向尾部
        if np.dot(direction, tail_center - head_center) < 0:
            direction *= -1
        print("方向向量:", direction)
        wire_features.append({
            'points': points,
            'direction': direction,
            'head': head_center,
            'tail': tail_center
        })
        
    plot_3d_wire_sections(wire_features, output_prefix+"_sections")
    #return [] # test
    # 步骤2：构建连接关系图
    connections = []
    for i in range(len(wire_features)):
        for j in range(i+1, len(wire_features)):
            wi = wire_features[i]
            wj = wire_features[j]
            
            # 计算四种可能的连接方式
            # Case 1: i的尾部连接j的头部
            d1, t1 = distance_to_ray(wj['head'], wi['tail'], wi['direction'])
            # Case 2: i的头部连接j的尾部, 射线起点为头部时, 射线方向相反
            d2, t2 = distance_to_ray(wj['tail'], wi['head'], -wi['direction'])
            # Case 3: j的尾部连接i的头部
            d3, t3 = distance_to_ray(wi['head'], wj['tail'], wj['direction'])
            # Case 4: j的头部连接i的尾部, 射线起点为头部时, 射线方向相反
            d4, t4 = distance_to_ray(wi['tail'], wj['head'], -wj['direction'])   
            # Case 5: i的尾部射线距离j的头部射线
            d5, t5, t6 = ray_distance(wi['tail'], wi['direction'], 
                             wj['head'], -wj['direction'])
            # Case 6: i的头部射线距离j的尾部射线
            d6, t7, t8  = ray_distance(wi['head'], -wi['direction'],
                             wj['tail'], wj['direction'])

            min_dist = min(d1, d2, d3, d4, d5, d6)
            # print(f"t1: {t1}, d1: {d1}")
            # print(f"t2: {t2}, d2: {d2}")
            # print(f"t3: {t3}, d3: {d3}")
            # print(f"t4: {t4}, d4: {d4}")
            # print(f"t5: {t5}, t6: {t6}, d5: {d5}")
            # print(f"t7: {t7}, t8: {t8}, d6: {d6}")      
            # print(f"min_dist: {min_dist}, dist_threshold: {dist_threshold}")    
            if min_dist < dist_threshold:
                connections.append((i, j, min_dist))
                print(f"连接关系: {i} -> {j}, 距离: {min_dist}")
        
    # 步骤3：分组，并排序
    wire_groups = []
    connection_map = defaultdict(list)  # 记录每个线段的连接关系

    # 建立双向连接图
    for i, j, _ in connections:
        connection_map[i].append(j)
        connection_map[j].append(i)

    # 标记已处理的线段
    processed = set()
    
    # 遍历所有线段进行分组
    for wire_id in range(len(wire_features)):
        if wire_id in processed:
            continue
            
        current_group = []
        queue = [wire_id]
        
        # 广度优先搜索收集相连线段
        while queue:
            current = queue.pop(0)
            if current in processed:
                continue
                
            current_group.append(wire_features[current])
            processed.add(current)
            
            # 添加所有相连且未处理的线段
            for neighbor in connection_map[current]:
                if neighbor not in processed:
                    queue.append(neighbor)
        
        # 在BFS收集线段后添加排序逻辑
        if len(current_group) >= 2:
            # 获取主方向（取第一个线段的方向）
            main_direction = current_group[0]['direction']
            main_point = current_group[0]['head']
            # 计算各线段中点到主射线的投影值
            projections = []
            for seg in current_group:
                # 计算线段中点
                mid_point = (seg['head'] + seg['tail']) / 2
                # 计算mid_point在主方向上的投影坐标
                projection = np.dot(mid_point - main_point, main_direction)
                projections.append(projection)
            
            # 按投影值排序
            current_group = [seg for _, seg in sorted(zip(projections, current_group))]

        wire_groups.append(current_group)    
        
    # 打印合并结果统计信息
    print("\n合并结果统计:")
    print(f"总组数: {len(wire_groups)}")
    for idx, group in enumerate(wire_groups):
        print(f"第 {idx+1} 组包含 {len(group)} 条线段")
        total_points = sum(len(seg['points']) for seg in group)
        print(f"  总点数: {total_points}")

    visualize_merged_wires(wire_groups, output_prefix+"_merged", b_draw_endpoints=False, b_draw_direction=False)
    
    return wire_groups

def check_occlusion(prev_seg, next_seg, all_wires, z_threshold=0.02, xy_thredhold=0.02, sample_step=0.001):
    """
    检测两个金线段之间是否存在遮挡
    参数:
        prev_seg: 前段金线特征字典
        next_seg: 后段金线特征字典
        all_wires: 所有已合并金线组的列表
        z_threshold: Z轴高度判定阈值(单位:mm)
    返回:
        bool: True表示存在遮挡，False表示无遮挡
    """
    print("遮挡判断:")
    # 1. 获取连接线段端点
    p1 = prev_seg['tail'].astype(np.float64)
    p2 = next_seg['head'].astype(np.float64)
    
    # 2. 生成连接线段采样点（XY平面投影）
    num_samples = max(100, int(np.linalg.norm(p2[:2] - p1[:2]) / sample_step))  # 每1um采样一个点
    t_values = np.linspace(0, 1, num_samples)
    connection_points = np.array([p1 + t*(p2-p1) for t in t_values])
    
    # 3. 构建连接线段参数方程
    def line_param(t):
        return p1 + t*(p2 - p1)
    
    # 4. 遍历所有其他金线
    for wire_idx, wire in enumerate(all_wires, 1):  # 金线组序号从1开始
        for seg_idx, seg in enumerate(wire, 1):     # 线段序号从1开始
            # 跳过当前正在连接的两个线段
            if seg is prev_seg or seg is next_seg:
                continue
                
            # 检查该线段的所有点
            for pt in seg['points']:
                pt = pt.astype(np.float64)
                
                # 4.1 在XY 平面，计算点到连接线段的投影参数
                vec = pt[:2] - p1[:2]
                dir_vec = p2[:2] - p1[:2]
                t = np.dot(vec, dir_vec) / (np.dot(dir_vec, dir_vec) + 1e-8)
                
                # 4.2 判断是否在投影区域内
                if 0 <= t <= 1: # 表示点投影在线段内部
                    # 计算投影点坐标
                    proj_pt = line_param(t)
                    xy_distance = np.linalg.norm(pt[:2] - proj_pt[:2])
                    
                    # 4.3 距离阈值判断
                    if xy_distance < xy_thredhold:
                        # 4.4 比较Z轴高度
                        connection_z = proj_pt[2]
                        if pt[2] > connection_z + z_threshold:
                            print(f"遮挡点位于: 第{wire_idx}组金线 第{seg_idx}段")
                            print(f"  遮挡点坐标: {np.round(pt, 3)}")
                            print(f"  对应连接线位置: {np.round(proj_pt, 3)} (距离值: {connection_z:.3f}mm)")
                            print(f"  垂直高度差: {pt[2] - connection_z:.3f}mm")
                            return True, pt
    print("---->无遮挡")
    return False, None

def cubic_hermite_point(p0, p1, m0, m1, t):
    """ 立方 Hermite 样条插值公式，返回曲线在参数 t ∈ [0,1] 处的坐标 """
    h00 =  2*t**3 - 3*t**2 + 1
    h10 =      t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 =      t**3 -    t**2
    return h00*p0 + h10*m0 + h01*p1 + h11*m1

def interpolate_segments_new(prev_seg, next_seg, step=0.002, tangent_scale=1.0):
    """
    使用立方 Hermite 样条，在 3D 空间平滑地连接两段:
      - 前段的尾点 = prev_seg['tail'], 方向 = prev_seg['direction']
      - 后段的头点 = next_seg['head'], 方向 = next_seg['direction']
    并返回插值后的点集(大小约为 num_samples+1)，
    保证曲线起点与前段尾点相同，终点与后段头点相同，且起止方向基本与给定向量对齐。

    参数:
      prev_seg: {'tail': np.array([x, y, z]), 'direction': np.array([dx, dy, dz])}
      next_seg: {'head': np.array([x, y, z]), 'direction': np.array([dx, dy, dz])}
      num_samples: 需要在 [0,1] 上采样多少份(实际输出点数是 num_samples+1)
      tangent_scale: 控制起止切线的长度 (默认=1.0)，可按需调大/调小来改变曲线弯曲程度
    """
    p0 = prev_seg['tail']                 # 前段尾点
    d0 = prev_seg['direction']            # 前段尾方向(建议传单位向量)
    p1 = next_seg['head']                 # 后段头点
    d1 = next_seg['direction']            # 后段头方向(建议传单位向量)

    # 如果两端点重合, 没什么可插值, 直接返回这个点
    if np.linalg.norm(p1 - p0) < 1e-12:
        return np.array([p0.copy()])

    # 根据需要, 决定如何给切线向量赋予“长度”
    # 一般会与 (p1 - p0) 距离相关, 也会与自己的方向相关.
    # 例如可以让:
    #   m0 = d0 * (||p1 - p0|| * tangent_scale)
    #   m1 = d1 * (||p1 - p0|| * tangent_scale)
    # 这样起止处的切线长度与首尾间距、以及一个缩放系数成正比.
    dist = np.linalg.norm(p1 - p0)
    m0 = d0 * (dist * tangent_scale)
    m1 = d1 * (dist * tangent_scale)

    # 逐步采样 t ∈ [0,1] 的插值点
    points = []
    direct_vector = p1 - p0  # 端点连线向量
    direct_length = np.linalg.norm(direct_vector)  # 端点间直线距离
    num_samples = int(direct_length/step)
    for i in range(num_samples+1):
        t = i / float(num_samples)  # 在 [0,1] 内均分
        pt = cubic_hermite_point(p0, p1, m0, m1, t)
        points.append(pt)
    return np.array(points)


def interpolate_golden_wires(merged_wires, step=0.002):
    """处理所有需要插值的金线段"""
    processed_wires = []
    occlusion_points = []
    for wire in merged_wires:
        if len(wire) <= 1:
            processed_wires.append(wire)
            continue
            
        new_wire = [wire[0]]
        for i in range(1, len(wire)):
            # 生成插值点
            b_occlusion, pt = check_occlusion(wire[i-1], wire[i], merged_wires)
            if 1:
                interpolated_points = interpolate_segments_new(wire[i-1], wire[i], step=0.002)
                occlusion_points.append(pt)

            # 创建插值线段特征
            if len(interpolated_points) > 0:
                interpolated_seg = {
                    'points': interpolated_points,
                    'direction': compute_pca_direction(interpolated_points),
                    'head': interpolated_points[0],
                    'tail': interpolated_points[-1]
                }
                new_wire.append(interpolated_seg)
            
            new_wire.append(wire[i])
        
        processed_wires.append(new_wire)
        # 打印合并结果统计信息
    print("\n插值后结果统计:")
    print(f"总组数: {len(processed_wires)}")
    for idx, group in enumerate(processed_wires):
        print(f"第 {idx+1} 组包含 {len(group)} 条线段")
        total_points = sum(len(seg['points']) for seg in group)
        print(f"  总点数: {total_points}")
    
    return processed_wires, occlusion_points

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
    
    # 4. 合并金线
    golden_wire_medlines_3d = golden_wire_medlines_3d[9:14]
    print(f"合并前金线数量: {len(golden_wire_medlines_3d)}")
    merged_golden_wires = merge_golden_wires(golden_wire_medlines_3d, 0.1, output_prefix)
    print(f"合并后金线数量: {len(merged_golden_wires)}")
    
    final_wires, occlusion_points = interpolate_golden_wires(merged_golden_wires, 0.001)
    visualize_merged_wires(final_wires, output_prefix+"_final", b_draw_endpoints=False, b_draw_direction=False, occlusion_points=occlusion_points)
    
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


def test_interpolate_segments():
    # 测试数据准备（添加显式类型声明）
    prev_seg = {
        'tail': np.array([1.0, 3.0, 0.0], dtype=np.float64),
        'head': np.array([-1.0, 3.0, 0.0], dtype=np.float64),
        'direction': np.array([1.0, 0.0, 0.0], dtype=np.float64),
    }

    next_seg = {
        'tail': np.array([3.0, -1.0, 0.0], dtype=np.float64),
        'head': np.array([3.0, 1.0, 0.0], dtype=np.float64),
        'direction': np.array([0.0, -1.0, 0.0], dtype=np.float64),
    }

    # 生成插值点
    interp_points = interpolate_segments_new(prev_seg, next_seg, 0.05, 1.0)

    print("插值点坐标:\n", interp_points)
    
    # 创建2D绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)  # 移除3D投影参数
    
    # 绘制插值点（移除z轴数据）
    ax.scatter(interp_points[:,0], interp_points[:,1], 
              c='r', marker='o', s=20, alpha=0.8, label='Interpolated Points')
    
    # 坐标轴设置（移除z轴标签）
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('2D Interpolated Points Visualization')
    
    # 添加图例（移除视角调整）
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"interpolation_test_2d.png", dpi=150)
    plt.close()
    
    return

if __name__ == "__main__":
    #main()
    test_interpolate_segments()


