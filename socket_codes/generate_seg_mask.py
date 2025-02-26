import os
import cv2
import numpy as np
import argparse
from skimage.morphology import skeletonize

def calc_mask_and_medline(txt_path, width, height, detect_method=3):
    """处理单个标注文件，返回中心线结果和坐标数据"""
    print(f"正在处理文件: {txt_path}")
    # 新建一个全是0的mask图
    mask = np.zeros((height, width), dtype=np.uint8)
    golden_wire_medlines = []
    medlines_mask = None  # 显式初始化为None
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"当前文件包含: {len(lines)} instances")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 按空格分割 (class_label, x1, y1, x2, y2, ...)
            coords = line.split()
            xy_vals = coords[1:]     # 从第二个数起为多边形坐标
            print(f"当前文件包含 {len(xy_vals)} 个坐标")
            if len(xy_vals) < 4:
                # 至少要有(x, y)一对才可绘制多边形
                continue
            
            # 将 0~1 归一化坐标转换为绝对坐标，然后打包成 [[x1, y1], [x2, y2], ...] 的形式
            polygon_points = []
            for i in range(0, len(xy_vals), 2):
                x_norm = float(xy_vals[i])
                y_norm = float(xy_vals[i+1])
                x = int(x_norm * width)
                y = int(y_norm * height)
                polygon_points.append([x, y])
            
            polygon_points = np.array(polygon_points, dtype=np.int32)
            
            # 用 fillPoly 在 mask 上把多边形区域填充为255
            cv2.fillPoly(mask, [polygon_points], 255)
            
            # ==== 新增中心线提取代码 ====
            # 创建单个实例的single_mask
            single_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(single_mask, [polygon_points], 255)
            if (detect_method == 1):
                # 方法一：基于骨架化（适用于简单形状）构建中心线
                # 骨架化处理
                medline_mask, sorted_points = calculate_centerline_by_skeleton(single_mask, is_by_distance_transform=False)         
            elif (detect_method == 2):
                # 方法二：基于中轴变换构建中心线
                medline_mask, sorted_points = calculate_centerline_by_skeleton(single_mask, is_by_distance_transform=True)
            else:
                # 方法三：基于主方向分析的中心线                    
                medline_mask, sorted_points = calculate_centerline_by_direction(single_mask)
                
            if medline_mask is not None:
                if medlines_mask is None:
                    medlines_mask = np.zeros_like(medline_mask)
                medlines_mask = cv2.bitwise_or(medlines_mask, medline_mask)
                
            golden_wire_medlines.append(sorted_points)
            
        # 创建彩色标记图（将单通道mask转为BGR）
        mask_with_medlines = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)     
        # 绘制每个中心线点
        for wire_idx, wire_points in enumerate(golden_wire_medlines):
            for (x, y) in wire_points:
                cv2.circle(mask_with_medlines, (int(x), int(y)), 3, [255, 0, 0], -1)
                
    return mask, mask_with_medlines, golden_wire_medlines

# 基于骨架化的中心线计算方法
def calculate_centerline_by_skeleton(single_mask, is_by_distance_transform=True):
    """基于中轴变换的中心线计算方法"""
    # 计算距离变换
    if is_by_distance_transform:
        dist_transform = cv2.distanceTransform(single_mask, cv2.DIST_L2, 3)
        _, medial_axis = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, cv2.THRESH_BINARY)
        medial_axis = np.uint8(medial_axis)
    else:
        medial_axis = single_mask
    
    # 骨架化处理
    skeleton = skeletonize(medial_axis//255)
    line_mask = np.uint8(skeleton * 255)
    
    # 获取非零坐标点
    y_coords, x_coords = np.where(line_mask > 0)
    if len(x_coords) == 0:
        return line_mask, []
    
    # 计算最小外接矩形
    rect = cv2.minAreaRect(np.column_stack((x_coords, y_coords)))
    (cx, cy), (w, h), angle = rect
    
    # 判断方向 (True: 横向, False: 纵向)
    is_horizontal = w > h
    print(f"当前对象是否水平: {is_horizontal}")
    # 按方向排序坐标点
    points = list(zip(x_coords, y_coords))
    if is_horizontal:
        # 按x坐标排序 (从左到右)
        sorted_points = sorted(points, key=lambda p: p[0])
    else:
        # 按y坐标排序 (从上到下)
        sorted_points = sorted(points, key=lambda p: p[1])
    
    return line_mask, sorted_points

# 基于主方向分析的中心线计算方法
def calculate_centerline_by_direction(single_mask):
    """基于主方向分析的中心线计算方法"""
    # 添加调试显示控制
    if not hasattr(calculate_centerline_by_direction, "has_displayed"):
        calculate_centerline_by_direction.has_displayed = 0
    
    # 获取非零像素坐标
    y_coords, x_coords = np.where(single_mask > 0)
    if len(x_coords) == 0:
        return None
    
    # 计算最小外接矩形
    x_min, y_min = np.min(x_coords), np.min(y_coords)
    x_max, y_max = np.max(x_coords), np.max(y_coords)
    roi = single_mask[y_min:y_max+1, x_min:x_max+1]
    
    # 计算缩放比例（保持长宽比）
    h, w = roi.shape[:2]

    # 首次调用时显示ROI和下采样结果
    if 0 and calculate_centerline_by_direction.has_displayed < 1:
        # 显示原始ROI区域
        cv2.imshow("First ROI", roi)        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        calculate_centerline_by_direction.has_displayed = 1
    
    # 计算中心点
    center_points = []
    if (0):
        # 下采样分析主方向    
        scale = 0.25
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        small_roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        if calculate_centerline_by_direction.has_displayed < 2:
            # 显示原始ROI区域
            cv2.imshow("Small ROI", small_roi)        
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            calculate_centerline_by_direction.has_displayed = 2
            
        # points = np.column_stack(np.where(small_roi > 0))
        points = np.column_stack(np.where(roi > 0))
        if len(points) < 2:
            return None
        # PCA主方向分析
        cov = np.cov(points.T)  # 计算协方差矩阵
        eig_vals, eig_vecs = np.linalg.eig(cov)  # 特征分解
        main_dir = eig_vecs[:, np.argmax(eig_vals)]  # 选择最大特征值对应的特征向量
        main_dir = main_dir / np.linalg.norm(main_dir)  # 单位化, 作为金线垂直方向
        # 计算golden_wire 的垂直角度（以度为单位）
        angle_rad = np.arctan2(main_dir[1], main_dir[0])  # 计算弧度
        angle_deg = np.degrees(angle_rad)  # 转换为角度
        print(f"金线垂直方向角度: {angle_deg:.2f}°")
        if calculate_centerline_by_direction.has_displayed < 3:
            print(f"主方向特征向量: {main_dir}, 类型: {main_dir.dtype}")
            print(f"最大特征值索引: {np.argmax(eig_vals)}, 对应特征值: {np.max(eig_vals):.2f}")
            calculate_centerline_by_direction.has_displayed = 3
                
        # 计算主方向的垂直方向向量
        perp_dir = np.array([main_dir[0], main_dir[1]]) 
        
        # 沿对角线生成采样点
        diag_start = np.array([0, 0])
        diag_end = np.array([w-1, h-1])
        num_samples = int(np.linalg.norm(diag_end - diag_start))  # 对角线长度
        t_values = np.linspace(0, 1, num_samples)
        
        # 在循环开始前创建调试图像
        debug_roi = cv2.cvtColor(roi.copy(), cv2.COLOR_GRAY2BGR)
        all_sample_pts = []  # 用于存储所有采样点坐标

        for t in t_values:
            # 沿对角线移动采样点
            sample_pt = (1-t)*diag_start + t*diag_end
            all_sample_pts.append(sample_pt)
                        
            # 沿主方向的垂直方向扫描
            scan_line = []
            for s in np.linspace(-1, 1, int(num_samples / 2)):
                x = int(sample_pt[0] + s*perp_dir[0]*max(w,h))
                y = int(sample_pt[1] + s*perp_dir[1]*max(w,h))
                
                if 0 <= x < w and 0 <= y < h:
                    if roi[y, x] > 0:
                        scan_line.append((x + x_min, y + y_min))  # 转换回原图坐标
            
            if scan_line:
                xs, ys = zip(*scan_line)
                center_x = int(np.mean(xs))
                center_y = int(np.mean(ys))
                center_points.append((center_x, center_y))
        
        # 计算所有采样点的均值中心
        if calculate_centerline_by_direction.has_displayed < 4:
            if all_sample_pts:
                for sample_pt in all_sample_pts:
                    pt = (int(sample_pt[0]), int(sample_pt[1]))
                    cv2.circle(debug_roi, pt, 5, (0,255,255), -1)
                
                for center_pt in center_points:
                    pt = (int(center_pt[0]-x_min), int(center_pt[1]-y_min))
                    cv2.circle(debug_roi, pt, 5, (0,0,255), -1)
                    
                # 计算均值点（ROI坐标系）
                mean_pt = np.mean(all_sample_pts, axis=0).astype(int)
                mean_x, mean_y = mean_pt

                # 绘制红色参考线（沿垂直方向延伸）
                line_length = max(w, h)
                start_pt = (int(mean_x - perp_dir[0]*line_length), 
                        int(mean_y - perp_dir[1]*line_length))
                end_pt = (int(mean_x + perp_dir[0]*line_length),
                        int(mean_y + perp_dir[1]*line_length))
                
                cv2.line(debug_roi, start_pt, end_pt, (0,0,255), 2, lineType=cv2.LINE_AA)
                cv2.circle(debug_roi, (mean_x, mean_y), 5, (0,255,255), -1)

                # 显示结果（缩放至合适尺寸）
                cv2.imshow("Sampling Visualization", debug_roi)
                cv2.waitKey(0)  # 显示3秒
                cv2.destroyAllWindows()
            calculate_centerline_by_direction.has_displayed = 4
    else:
        is_horizontal = w > h
        print(f"主方向是否水平: {is_horizontal}")
        if is_horizontal:
            for col in range(x_min, x_max+1):
                ys = np.where(single_mask[:, col] > 0)[0]
                if ys.size > 0:
                    center_points.append((col, int(np.mean(ys))))
        else:
            for row in range(y_min, y_max+1):
                xs = np.where(single_mask[row, :] > 0)[0]
                if xs.size > 0:
                    center_points.append((int(np.mean(xs)), row))
    

    
    # 生成中心线
    if not center_points:
        return None, None
    
    line_mask = np.zeros_like(single_mask)
    for i in range(1, len(center_points)):
        cv2.line(line_mask, center_points[i-1], center_points[i], 255, 1)
    
    return line_mask, center_points

def generate_seg_masks(
    input_dir="./golden_wire/labels",
    output_dir="./golden_wire/masks",
    width=5120,
    height=5120
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历所有.txt文件
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".txt"):
            txt_path = os.path.join(input_dir, file_name)
            
            # 调用处理函数
            mask, medlines_mask, golden_wire_medlines = calc_mask_and_medline(txt_path, width, height, detect_method=3)

            # 保存结果
            out_name = file_name.replace(".txt", "_mask.jpg")
            cv2.imwrite(os.path.join(output_dir, out_name), mask)
            
            centerline_name = file_name.replace(".txt", "_medline_mask.jpg")
            cv2.imwrite(os.path.join(output_dir, centerline_name), medlines_mask)      
    return 

def main():
    """
    带有命令行参数的入口函数，可指定输入/输出目录及图像尺寸等信息。
    """
    input_dir = "./golden_wire/labels"
    output_dir = "./golden_wire/labels"
    width = 5120
    height = 5120
    
    generate_seg_masks(
        input_dir=input_dir,
        output_dir=output_dir,
        width=width,
        height=height
    )

if __name__ == "__main__":
    main()
