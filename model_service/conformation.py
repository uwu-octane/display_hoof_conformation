import math
import re
from datetime import datetime
import pandas as pd
import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from shapely import Point
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend if running in a non-GUI environment

import numpy as np
from scipy.spatial import ConvexHull


import torchvision.transforms.v2 as transforms


def get_diagonals(polygon):
    if isinstance(polygon, Polygon) and len(polygon.exterior.coords) == 5:  # 四个顶点加上首尾闭合的点
        # 提取顶点
        coords = list(polygon.exterior.coords[:-1])  # 忽略闭合的重复点
        # 创建对角线
        diagonal1 = LineString([coords[0], coords[2]])
        diagonal2 = LineString([coords[1], coords[3]])
        return diagonal1, diagonal2
    else:
        return None


def get_simplified_poly(edge_poly):
    original_polygon = Polygon(edge_poly)
    # polygon_boundary = original_polygon.boundary
    tolerance = 50
    simplified_polygon = original_polygon.simplify(tolerance)
    return simplified_polygon


# def get_vm_simplified_poly(edge_poly):
#     polygon = get_simplified_poly(edge_poly)
#     vertices = list(polygon.exterior.coords)
#     simplifier = vw.Simplifier(vertices)
#     sample_poly = simplifier.simplify(number=4)
#     sample_poly = Polygon(sample_poly)
#     return sample_poly


def draw_siplified_poly(image_np, simplified_poly, path=None):
    x_sim, y_sim = simplified_poly.exterior.xy
    copy = image_np.copy()
    plt.scatter(x_sim, y_sim, color='blue', s=2)
    plt.plot(x_sim, y_sim, 'r-', label='Simplified Polygon', lw=0.5)
    plt.plot(x_sim, y_sim, 'r-', label='Simplified Polygon', lw=0.5)
    # plt.plot([x_sim[-1], x_sim[0]], [y_sim[-1], y_sim[0]], 'r-')
    plt.axis('off')
    plt.imshow(copy)
    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()


def draw_contours(image_np, edge_poly, path=None, background=False):
    x_sim, y_sim = edge_poly.exterior.xy
    if background:
        copy = image_np.copy()
    else:
        copy = np.ones_like(image_np) * 255

    fig, ax = plt.subplots()
    ax.imshow(copy)
    ax.plot(x_sim, y_sim, 'r-', label='Simplified Polygon', lw=0.5)
    ax.axis('off')

    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = Image.frombuffer('RGBA', fig.canvas.get_width_height(), buf, 'raw', 'RGBA', 0, 1)

    plt.close(fig)

    if path is not None:
        img.save(path, format='PNG')

    return img


def draw_contours_with_diagonal(image_np, simplified_poly, diagonal1, diagonal2, path=None, centroid=None):
    x_sim, y_sim = simplified_poly.exterior.xy
    copy = np.zeros_like(image_np)
    # plt.scatter(x_sim, y_sim, color='blue')
    plt.plot(x_sim, y_sim, 'r-', label='Simplified Polygon')
    plt.plot(x_sim, y_sim, 'r-', label='Simplified Polygon')
    x1, y1 = diagonal1.xy
    # plt.plot(x1, y1, 'bo-', label='Diagonal 1', markersize=2)  # 用红色标记对角线1
    x2, y2 = diagonal2.xy
    # plt.plot(x2, y2, 'bo-', label='Diagonal 2', markersize=2)  # 用蓝色标记对角线2
    plt.imshow(copy)
    # plt.plot([x_sim[-1], x_sim[0]], [y_sim[-1], y_sim[0]], 'r-')
    if centroid is not None:
        plt.scatter(centroid.x, centroid.y, color='green', label='Centroid', s=2)

    plt.axis('off')
    if path is None:
        plt.show()
    else:
        # plt.gca().invert_yaxis()
        plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()


def draw_diagonal(image_np, diagonal1, diagonal2, path=None, centroid=None):
    copy = image_np.copy()
    plt.imshow(copy)
    # diagonal1, diagonal2 = get_diagonals(simplified_polygon)
    x1, y1 = diagonal1.xy
    plt.plot(x1, y1, 'bo-', label='Diagonal 1', markersize=5)  # 用红色标记对角线1
    x2, y2 = diagonal2.xy
    plt.plot(x2, y2, 'bo-', label='Diagonal 2', markersize=5)  # 用蓝色标记对角线2

    if centroid is not None:
        plt.scatter(centroid.x, centroid.y, color='green', label='Centroid', s=5)
    plt.axis('off')
    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()


def classify_vertices(polygon, diagonal1, diagonal2):
    def get_line_params(line):
        x1, y1, x2, y2 = *line.coords[0], *line.coords[1]
        slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
        intercept = y1 - slope * x1
        return slope, intercept

    def position_relative_to_line(x, y, slope, intercept):
        # 线方程为 y = slope * x + intercept
        # 如果点的 y 坐标大于该 x 坐标在直线上的 y 坐标，则点在线上方，否则在下方
        y_line = slope * x + intercept
        if np.isnan(y_line) or np.isinf(y_line):
            return np.nan
        return np.sign(y - y_line)  # 返回点在线的上方（+1），线上（0），或下方（-1）

    # 获取对角线的参数
    slope1, intercept1 = get_line_params(diagonal1)
    slope2, intercept2 = get_line_params(diagonal2)

    # 初始化子集
    subsets = {'top': [], 'left': [], 'bottom': [], 'right': []}

    # 分类顶点
    for x, y in np.array(polygon.exterior.coords)[:-1]:  # 排除重复的起始顶点
        with np.errstate(invalid='ignore'):
            pos1 = position_relative_to_line(x, y, slope1, intercept1)
            pos2 = position_relative_to_line(x, y, slope2, intercept2)
            if np.isnan(pos1) or np.isnan(pos2):
                return None

            # 确定每个点的位置分类
            if pos1 > 0 and pos2 > 0:
                subsets['bottom'].append((x, y))
            elif pos1 > 0 > pos2:
                subsets['left'].append((x, y))
            elif pos1 < 0 and pos2 < 0:
                subsets['top'].append((x, y))
            elif pos1 < 0 < pos2:
                subsets['right'].append((x, y))

    return subsets


def fit_lines_to_subsets(subsets, weighted=True):
    line_params = {}

    for key, vertices in subsets.items():
        # 排序顶点
        sorted_vertices = sorted(vertices, key=lambda v: v[0])
        # 移除最左和最右的点
        if len(sorted_vertices) > 2:  # 确保有足够的点进行操作
            sorted_vertices = sorted_vertices[1:-1]
        else:
            continue  # 如果不足以移除两个点，则跳过

        # 提取x和y坐标
        x, y = zip(*sorted_vertices)
        if weighted:
            # if key == 'top':
            m, c, x_sorted, y_sorted, weights = weighted_fit_with_center_low_weight(sorted_vertices, 10, 0.5)
            line_params[key] = {'slope': m, 'intercept': c, 'weights': weights}
        # m, c, x_sorted, y_sorted, weights = weighted_fit_with_center_low_weight(sorted_vertices)
        # line_params[key] = {'slope': m, 'intercept': c, 'weights': weights}

        if len(x) > 1:  # 确保至少有两个点进行拟合
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            line_params[key] = {'slope': m, 'intercept': c}

    return line_params


def draw_lines_on_image(image_np, line_params, path=None):
    """
    在图像上绘制直线。

    参数:
        image_np (numpy.ndarray): 输入图像。
        line_params (dict): 包含直线参数的字典。
            示例: {'top': {'slope': m, 'intercept': c}}
    """
    copy = image_np.copy()

    for key, params in line_params.items():
        slope = params['slope']
        intercept = params['intercept']

        # 计算图像边界上的点
        y1 = intercept
        y2 = slope * image_np.shape[1] + intercept
        pt1 = (0, int(y1))
        pt2 = (image_np.shape[1], int(y2))
        # 绘制直线
        cv2.line(copy, pt1, pt2, (0, 255, 0), 2)
    plt.axis('off')
    plt.imshow(copy)
    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()


def weighted_fit_with_center_low_weight(points, center_weight=8, center_ratio=0.7):
    # 从点集中提取x和y坐标
    x = np.array([point[0] for point in points])
    y = np.array([point[1] for point in points])
    # 排序数据点
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    # 确定中间20%的点的索引范围
    num_points = len(x_sorted)
    middle_start = int(num_points * (1 - center_ratio) / 2)
    middle_end = int(num_points * (1 + center_ratio) / 2)
    # 创建权重数组
    weights = np.ones(num_points)
    weights[middle_start:middle_end] = center_weight  # 中间20%的点权重较小

    # 进行加权线性拟合
    params = np.polyfit(x_sorted, y_sorted, 1, w=weights)
    m, c = params

    return m, c, x_sorted, y_sorted, weights


def draw_group_points(image_np, edge_poly, path=None):
    original_polygon = Polygon(edge_poly)
    simplified_polygon = Polygon(simplify_with_hull(edge_poly))
    diagonal1, diagonal2 = get_diagonals(simplified_polygon)
    subsets = classify_vertices(original_polygon, diagonal1, diagonal2)
    colors = ['blue', 'green', 'red', 'purple']
    copy = image_np.copy()

    plotted_points = set()

    # Create a figure and axis without showing it
    fig, ax = plt.subplots()
    ax.imshow(copy)
    ax.axis('off')

    for idx, (key, vertices) in enumerate(subsets.items()):
        if vertices:  # Check if not empty
            vx, vy = zip(*vertices)
            subset_points = set(vertices)  # Convert to set for easy comparison
            unique_points = subset_points - plotted_points  # Only plot unique points

            if unique_points:
                vx, vy = zip(*unique_points)
                plt.scatter(vx, vy, color=colors[idx], label=f'Subset {key}', s=10)
                plotted_points.update(unique_points)  # Add to plotted points

            print(key + '_' + colors[idx])
        # Convert the Matplotlib figure to a PIL Image
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = Image.frombuffer('RGBA', fig.canvas.get_width_height(), buf, 'raw', 'RGBA', 0, 1)
    plt.close(fig)  # Close the figure to avoid displaying it

    if path is not None:
        img.save(path, format='PNG')

    return img


def find_intersection(line1, line2):
    m1, c1 = line1['slope'], line1['intercept']
    m2, c2 = line2['slope'], line2['intercept']
    w1 = line1.get('weights', 1)
    w2 = line2.get('weights', 1)

    if m1 == m2:
        return None  # Parallel lines, no intersection

    # 计算交点
    x = (c2 - c1) / (m1 - m2)
    y = m1 * x + c1

    # 如果有权重，计算加权平均
    if w1 is not None and w2 is not None:
        if isinstance(w1, (np.ndarray, list)):
            w1 = np.mean(w1)
        if isinstance(w2, (np.ndarray, list)):
            w2 = np.mean(w2)
        x_weighted = (w1 * x + w2 * x) / (w1 + w2)
        y_weighted = (w1 * y + w2 * (m2 * x + c2)) / (w1 + w2)
        return (x_weighted, y_weighted)

    return (x, y)


def get_result_intersections(line_params):
    top = line_params['top']
    left = line_params['left']
    right = line_params['right']
    bottom = line_params['bottom']
    p1 = find_intersection(top, left)
    p2 = find_intersection(left, bottom)

    p3 = find_intersection(bottom, right)
    p4 = find_intersection(right, top)
    points = [p1, p2, p3, p4]
    # 按x坐标排序
    points_sorted_by_x = sorted(points, key=lambda p: p[0])

    # 分成左右两部分
    left_points = points_sorted_by_x[:2]
    right_points = points_sorted_by_x[2:]

    # 分别对左半部分和右半部分按y坐标排序
    left_points_sorted_by_y = sorted(left_points, key=lambda p: p[1])
    right_points_sorted_by_y = sorted(right_points, key=lambda p: p[1])

    # 分别确定左上、左下、右上、右下
    left_bottom = left_points_sorted_by_y[0]
    left_top = left_points_sorted_by_y[1]
    right_bottom = right_points_sorted_by_y[0]
    right_top = right_points_sorted_by_y[1]

    intersections = {'left_top': left_bottom, 'left_bottom': left_top, 'right_bottom': right_top,
                     'right_top': right_bottom}
    return intersections


def draw_intersections(image_np, intersections, path=None):
    left_top = intersections['left_top']
    left_bottom = intersections['left_bottom']
    right_bottom = intersections['right_bottom']
    right_top = intersections['right_top']
    copy = image_np.copy()
    plt.scatter(left_top[0], left_top[1], color='blue', s=10)
    plt.scatter(left_bottom[0], left_bottom[1], color='green', s=10)
    plt.scatter(right_bottom[0], right_bottom[1], color='red', s=10)
    plt.scatter(right_top[0], right_top[1], color='purple', s=10)
    plt.axis('off')
    plt.imshow(copy)
    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()


def get_result_poly(intersections):
    left_top = intersections['left_top']
    left_bottom = intersections['left_bottom']
    right_bottom = intersections['right_bottom']
    right_top = intersections['right_top']
    poly = Polygon([left_top, left_bottom, right_bottom, right_top])
    return poly


def vector_magnitude(vector):
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2)


def point_to_vector_distance(point, start_point, vector):
    # 计算点到向量起点的向量
    new_vec = point - start_point
    unit = vector / np.linalg.norm(vector)
    project_length = np.dot(new_vec, unit)
    project_point = start_point + project_length * unit
    distance = np.linalg.norm(point - project_point)
    return distance


def angle_between_vectors(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)

    # 计算夹角的余弦值
    cos_theta = dot_product / (norm_A * norm_B)

    # 计算夹角（弧度）
    angle_radians = np.arccos(cos_theta)

    # 将弧度转换为角度
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


def get_conformation(intersections, length, normalized=True):
    left_top = intersections['left_top']
    left_bottom = intersections['left_bottom']
    right_bottom = intersections['right_bottom']
    right_top = intersections['right_top']
    left_line_length = euclidean_distance(left_top, left_bottom)
    right_line_length = euclidean_distance(right_bottom, right_top)

    left_top = np.array([left_top[0], left_top[1]])
    left_bottom = np.array([left_bottom[0], left_bottom[1]])
    right_bottom = np.array([right_bottom[0], right_bottom[1]])
    right_top = np.array([right_top[0], right_top[1]])

    if left_line_length > right_line_length:
        # dorsal_hoof_wall_vector = (left_top[0] - left_bottom[0], left_top[1] - left_bottom[1])
        # weight_bearing_vector = (right_bottom[0] - left_bottom[0], right_bottom[1] - left_bottom[1])
        # coronary_band_vector = (right_top[0] - left_top[0], right_top[1] - left_top[1])
        # heel_vector = (right_top[0] - right_bottom[0], right_top[1] - right_bottom[1])
        dorsal_hoof_wall_vector = left_top - left_bottom
        weight_bearing_vector = right_bottom - left_bottom
        coronary_band_vector = right_top - left_top
        heel_vector = right_top - right_bottom

        heel_height = point_to_vector_distance(right_top, left_bottom, weight_bearing_vector)

        dorsal_coronary_band_height = point_to_vector_distance(left_top, left_bottom,
                                                               weight_bearing_vector)
    else:
        # dorsal_hoof_wall_vector = (right_bottom[0] - right_top[0], right_bottom[1] - right_top[1])
        # weight_bearing_vector = (left_bottom[0] - right_bottom[0], left_bottom[1] - right_bottom[1])
        # coronary_band_vector = (left_top[0] - right_top[0], left_top[1] - right_top[1])
        # heel_vector = (left_top[0] - left_bottom[0], left_top[1] - left_bottom[1])
        dorsal_hoof_wall_vector = right_top - right_bottom
        weight_bearing_vector = left_bottom - right_bottom
        coronary_band_vector = left_top - right_top
        heel_vector = left_top - left_bottom

        heel_height = point_to_vector_distance(left_top, right_bottom, weight_bearing_vector)
        dorsal_coronary_band_height = point_to_vector_distance(right_top, right_bottom,
                                                               weight_bearing_vector)

    dorsal_hoof_wall_length = np.linalg.norm(dorsal_hoof_wall_vector)
    weight_bearing_length = np.linalg.norm(weight_bearing_vector)

    dorsal_hoof_wall_angle = angle_between_vectors(dorsal_hoof_wall_vector, weight_bearing_vector)
    coronary_band_angle = angle_between_vectors(coronary_band_vector, weight_bearing_vector)
    heel_angle = angle_between_vectors(heel_vector, weight_bearing_vector)
    perimeter = length
    normalized_dorsal_hoof_wall_length = dorsal_hoof_wall_length / perimeter
    normalized_weight_bearing_length = weight_bearing_length / perimeter
    normalized_heel_height = heel_height / perimeter
    normalized_dorsal_coronary_band_height = dorsal_coronary_band_height / perimeter

    conformation = {'dorsal_hoof_wall_length': dorsal_hoof_wall_length, 'weight_bearing_length': weight_bearing_length,
                    'heel_height': heel_height, 'dorsal_coronary_band_height': dorsal_coronary_band_height,
                    'dorsal_hoof_wall_angle': dorsal_hoof_wall_angle, 'coronary_band_angle': coronary_band_angle,
                    'heel_angle': heel_angle}
    if normalized:
        conformation['dorsal_hoof_wall_length'] = normalized_dorsal_hoof_wall_length
        conformation['weight_bearing_length'] = normalized_weight_bearing_length
        conformation['heel_height'] = normalized_heel_height
        conformation['dorsal_coronary_band_height'] = normalized_dorsal_coronary_band_height
    return conformation


def process_mask(mask_uint8, kernel_size=3, iterations=1):
    height = mask_uint8.shape[0]
    kernel = height // 10 + 11
    if kernel % 2 == 0:
        kernel += 1
    blurred = cv2.GaussianBlur(mask_uint8, (kernel, kernel), 0)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilation = cv2.dilate(blurred, kernel, iterations=iterations)
    erosion = cv2.erode(blurred, kernel, iterations=iterations)

    # 计算边缘：膨胀图像与侵蚀图像之差
    edges = dilation - erosion

    return edges


def get_contours(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [
        np.squeeze(contour, axis=1)
        for contour in contours
        if contour.shape[0] >= 3
    ]


def convert_contours_to_Polygon(contours):
    if len(contours) == 0:
        return None
    else:
        contours = contours[0]
        return Polygon(contours)


def get_conformation_from_mask(mask, weighted=False):
    mask_uint8 = mask.astype(np.uint8)
    height, width = mask_uint8.shape

    edges = process_mask(mask_uint8, height, kernel_size=3, iterations=1)

    contours = get_contours(edges)

    edges_in_poly = contours[0]
    original_polygon = Polygon(edges_in_poly)

   # simplified_polygon = get_vm_simplified_poly(edges_in_poly)
    simplified_polygon = None
    diagonal1, diagonal2 = get_diagonals(simplified_polygon)

    subsets = classify_vertices(original_polygon, diagonal1, diagonal2)

    line_params = fit_lines_to_subsets(subsets, weighted)
    intersection = get_result_intersections(line_params)

    # poly = get_result_poly(intersection)
    conformation = get_conformation(intersection)

    return conformation, intersection, simplified_polygon


def perpendicular_intersection(point, line):
    # 获取线段的起点和终点
    p1 = np.array(line.coords[0])
    p2 = np.array(line.coords[1])

    # 转换点
    p = np.array(point.coords[0])

    # 计算线段向量
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len

    # 计算点到线段起点的向量
    p_vec = p - p1

    # 投影
    proj_length = np.dot(p_vec, line_unitvec)
    proj_point = p1 + proj_length * line_unitvec

    return Point(proj_point)


def plot_conformation(intersections, conformation, image, path=None):
    left_top = intersections['left_top']
    left_bottom = intersections['left_bottom']
    right_bottom = intersections['right_bottom']
    right_top = intersections['right_top']
    left_length = euclidean_distance(left_top, left_bottom)
    right_length = euclidean_distance(right_bottom, right_top)
    img = image.copy()

    fig, ax = plt.subplots()
    ax.imshow(img)

    # 计算垂线
    bottom_line = LineString([left_bottom, right_bottom])

    # 绘制轮廓
    ax.plot([left_top[0], right_top[0]], [left_top[1], right_top[1]], 'r-', linewidth=2.5)  # top edge
    ax.plot([left_bottom[0], right_bottom[0]], [left_bottom[1], right_bottom[1]], 'r-', linewidth=2.5)  # bottom edge
    ax.plot([left_top[0], left_bottom[0]], [left_top[1], left_bottom[1]], 'r-', linewidth=2.5)  # left edge
    ax.plot([right_top[0], right_bottom[0]], [right_top[1], right_bottom[1]], 'r-', linewidth=2.5)  # right edge

    # 标注测量值
    if left_length > right_length:
        intersection = perpendicular_intersection(Point(left_top), bottom_line)
        ax.plot([left_top[0], intersection.x], [left_top[1], intersection.y], 'r-', linewidth=3)

        ax.annotate(f"A: {conformation['dorsal_hoof_wall_length']:.2f}",
                    xy=(left_top[0], left_top[1]),
                    xytext=(left_bottom[0] - 50, (left_top[1] + left_bottom[1]) / 2), color='blue')

        ax.annotate(f"C: {conformation['heel_height']:.2f}",
                    xy=(right_top[0], right_top[1]),
                    xytext=((right_bottom[0] + right_top[0]) / 2 + 70, (right_top[1] + right_bottom[1]) / 2),
                    color='blue')

        ax.annotate(f"G: {conformation['dorsal_coronary_band_height']:.2f}",
                    xy=(left_top[0], left_top[1]), xytext=(left_top[0], left_top[1] / 2 + left_top[1]), color='blue')

        ax.annotate(f"D: {conformation['dorsal_hoof_wall_angle']:.2f}°",
                    xy=(left_bottom[0], left_bottom[1]), xytext=(left_bottom[0] - 40, left_bottom[1] + 50),
                    color='blue')

        ax.annotate(f"E: {conformation['coronary_band_angle']:.2f}°",
                    xy=(right_top[0], right_top[1]), xytext=(right_top[0] + 40, right_top[1] - 30), color='blue')

        ax.annotate(f"F: {conformation['heel_angle']:.2f}°",
                    xy=(right_bottom[0], right_bottom[1]), xytext=(right_bottom[0] + 40, right_bottom[1] + 30),
                    color='blue')

    else:
        intersection = perpendicular_intersection(Point(right_top), bottom_line)
        ax.plot([right_top[0], intersection.x], [right_top[1], intersection.y], 'r-', linewidth=3)
        ax.annotate(f"A: {conformation['dorsal_hoof_wall_length']:.2f}",
                    xy=(right_top[0], right_top[1]),
                    xytext=((right_bottom[0] + right_top[0]) / 2 + 100, (right_top[1] + right_bottom[1]) / 2),
                    color='blue')

        ax.annotate(f"C: {conformation['heel_height']:.2f}",
                    xy=(left_top[0], left_top[1]),
                    xytext=((left_top[0] + left_bottom[0]) / 2 - 300, (left_top[1] + left_bottom[1]) / 2 + 50),
                    color='blue')

        ax.annotate(f"G: {conformation['dorsal_coronary_band_height']:.2f}",
                    xy=(right_top[0], right_top[1]), xytext=(right_top[0], (right_top[1] + intersection.y) / 2),
                    color='blue')

        ax.annotate(f"D: {conformation['dorsal_hoof_wall_angle']:.2f}°",
                    xy=(right_bottom[0], right_bottom[1]), xytext=(right_bottom[0] + 40, right_bottom[1] + 50),
                    color='blue')

        ax.annotate(f"E: {conformation['coronary_band_angle']:.2f}°",
                    xy=(left_top[0], left_top[1]), xytext=(left_top[0] - 50, left_top[1] - 30), color='blue')

        ax.annotate(f"F: {conformation['heel_angle']:.2f}°",
                    xy=(left_bottom[0], left_bottom[1]), xytext=(left_bottom[0] - 40, left_bottom[1] + 70),
                    color='blue')
    ax.annotate(f"B: {conformation['weight_bearing_length']:.2f}",
                xy=(left_bottom[0], left_bottom[1]),
                xytext=((left_bottom[0] + right_bottom[0]) / 2, (left_bottom[1] + right_bottom[1]) / 2 + 70),
                color='blue')

    ax.set_aspect('equal')
    plt.axis('off')
    # Convert the Matplotlib figure to a PIL Image object
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img_pil = Image.frombuffer('RGBA', fig.canvas.get_width_height(), buf, 'raw', 'RGBA', 0, 1)

    plt.close(fig)  # Close the figure to free up memory

    if path is not None:
        img_pil.save(path, format='PNG')

    return img_pil


def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def simplify_with_hull(original_polygon, draw=None):
    points = np.array(original_polygon.exterior.coords)

    hull = ConvexHull(points, incremental=True)

    # 提取凸包顶点
    hull_points = points[hull.vertices]

    max_distance = 0
    longest_diagonal = (None, None)

    num_points = len(hull_points)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            dist = euclidean_distance(hull_points[i], hull_points[j])
            if dist > max_distance:
                max_distance = dist
                longest_diagonal = (hull_points[i], hull_points[j])

    # 找到最左上的端点
    leftmost_point = min(hull_points, key=lambda p: p[0])
    rightmost_point = max(hull_points, key=lambda p: p[0])
    left_top = min(hull_points, key=lambda p: p[0] + p[1])
    right_bottom = max(hull_points, key=lambda p: p[0] + p[1])

    # 分割点集
    left_group = []
    right_group = []

    # 函数计算点相对于直线的位置
    def is_above_line(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

    for point in points:
        if is_above_line(point, longest_diagonal[0], longest_diagonal[1]) > 0:
            left_group.append(point)
        else:
            right_group.append(point)

    left_group = np.array(left_group)
    right_group = np.array(right_group)

    # 函数计算点到直线的距离
    def distance_to_line(p, a, b):
        return np.abs((b[1] - a[1]) * p[0] - (b[0] - a[0]) * p[1] + b[0] * a[1] - b[1] * a[0]) / np.sqrt(
            (b[1] - a[1]) ** 2 + (b[0] - a[0]) ** 2)

    # 找到每组中距离分割线最远的点
    if len(left_group) > 0:
        furthest_left_point = max(left_group, key=lambda p: distance_to_line(p, leftmost_point, rightmost_point))
    else:
        furthest_left_point = None

    if len(right_group) > 0:
        furthest_right_point = max(right_group, key=lambda p: distance_to_line(p, leftmost_point, rightmost_point))
    else:
        furthest_right_point = None
    if draw != None:
        draw = draw.copy()
        plt.scatter(left_top[0], left_top[1], color='blue', label='Leftmost Point')
        plt.scatter(right_bottom[0], right_bottom[1], color='green', label='Rightmost Point')
        plt.plot([left_top[0], right_bottom[0]], [left_top[1], right_bottom[1]], 'k-', lw=1
                 , label='Division Line')
        plt.scatter(left_group[:, 0], left_group[:, 1], color='purple', label='Upper Group', s=5)
        plt.scatter(right_group[:, 0], right_group[:, 1], color='orange', label='Lower Group', s=5)
        if furthest_left_point is not None:
            plt.scatter(furthest_left_point[0], furthest_left_point[1], color='cyan', label='Furthest Left Points')

        if furthest_right_point is not None:
            plt.scatter(furthest_right_point[0], furthest_right_point[1], color='magenta',
                        label='Furthest Right Points')

        plt.axis('off')
        plt.imshow(draw)
    return [longest_diagonal[0], furthest_left_point, longest_diagonal[1], furthest_right_point]


# def get_mask(result, original_image_size, min_score_thresh=0.6):
#     detection_masks = tf.convert_to_tensor(result['detection_masks'][0])
#     detection_scores = tf.convert_to_tensor(result['detection_scores'][0])
#     detection_boxes = tf.convert_to_tensor(result['detection_boxes'][0])
#     max_score_index = tf.argmax(detection_scores).numpy()
#
#     best_mask = detection_masks[max_score_index]
#     best_box = detection_boxes[max_score_index]
#
#     best_mask = tf.expand_dims(best_mask, axis=0)
#     best_box = tf.expand_dims(best_box, axis=0)
#
#     detection_masks_reframed = reframe_box_masks_to_image_masks(
#         best_mask, best_box / 640.0,
#         original_image_size[0], original_image_size[1])
#
#     detection_masks_reframed = tf.cast(detection_masks_reframed > min_score_thresh, np.uint8)
#
#     mask = detection_masks_reframed[0].numpy()
#     mask_uint8 = mask.astype(np.uint8)
#
#     return mask_uint8


# def build_input_for_keydet(image_np, contours_poly):
#     copy = np.zeros_like(image_np)
#     centroid = contours_poly.centroid
#     x_sim, y_sim = contours_poly.exterior.xy
#     pts = np.array([[int(x), int(y)] for x, y in zip(x_sim, y_sim)], np.int32)
#     pts = pts.reshape((-1, 1, 2))
#     cv2.polylines(copy, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
#
#     if centroid is not None:
#         cv2.circle(copy, (int(centroid.x), int(centroid.y)), 2, (0, 255, 0), -1)
#
#     # Convert numpy array to PIL Image
#     copy = cv2.cvtColor(copy, cv2.COLOR_BGR2RGB)
#     img_pil = Image.fromarray(copy).convert('RGB')
#     # plt.axis('off')
#     # plt.imshow(img_pil)
#     input_image_keydet = resize_img(img_pil, target_sz=640, divisor=1)
#     input_tensor = transforms.Compose([transforms.ToImage(),
#                                        transforms.ToDtype(torch.float32, scale=True)])(input_image_keydet)[None]
#     return input_tensor


def get_keypoints(result, min_img_scale):
    scores_cpu = result['scores'].cpu()
    if scores_cpu.numel() == 0:
        print("The scores tensor is empty.")
        return None
    max_score_index = np.argmax(scores_cpu)

    # Create a mask with False for all entries except the one with the highest score
    scores_mask = np.zeros_like(scores_cpu, dtype=bool)
    scores_mask[max_score_index] = True
    # scores_mask = result['scores'] > conf_threshold
    predicted_keypoints = result['keypoints'].cpu()
    predicted_keypoints = (predicted_keypoints[scores_mask])[:, :, :-1].reshape(-1, 2) * min_img_scale
    predicted_keypoints = predicted_keypoints[:5]
    if len(predicted_keypoints) < 5:
        return None
    toe_point = predicted_keypoints[3]
    heel_a_point = predicted_keypoints[1]
    heel_c_point = predicted_keypoints[2]
    coronary_band_point = predicted_keypoints[4]
    return {'toe_point': toe_point, 'heel_a_point': heel_a_point, 'heel_c_point': heel_c_point,
            'coronary_band_point': coronary_band_point}


def get_keypoints_in_Poly(result, min_img_scale):
    # scores_cpu = result['scores'].cpu()
    # if scores_cpu.numel() == 0:
    #     print("The scores tensor is empty.")
    #     return None
    # max_score_index = np.argmax(scores_cpu)
    #
    # # Create a mask with False for all entries except the one with the highest score
    # scores_mask = np.zeros_like(scores_cpu, dtype=bool)
    # scores_mask[max_score_index] = True
    # #scores_mask = result['scores'] > conf_threshold
    # predicted_keypoints = result['keypoints'].cpu()
    # predicted_keypoints = (predicted_keypoints[scores_mask])[:, :, :-1].reshape(-1, 2) * min_img_scale
    # predicted_keypoints = predicted_keypoints[:5]
    # if len(predicted_keypoints) < 5:
    #     return None
    # toe_point = predicted_keypoints[2]
    # heel_a_points = predicted_keypoints[4]
    # heel_c_points = predicted_keypoints[3]
    # coronary_band_points = predicted_keypoints[1]
    keypoints = get_keypoints(result, min_img_scale)
    toe_point = keypoints['toe_point']
    heel_a_points = keypoints['heel_a_point']
    heel_c_points = keypoints['heel_c_point']
    coronary_band_points = keypoints['coronary_band_point']
    poly = Polygon([toe_point, heel_c_points, heel_a_points, coronary_band_points])
    return poly


def get_conformation_with_keydet(contours_poly, keypoints_poly, weighted=False, normalized=True):
    if keypoints_poly is None:
        print('no keypoints')
        return None
    if get_diagonals(keypoints_poly) is None:
        print('diagonal failed')
        return None
    diagonal1, diagonal2 = get_diagonals(keypoints_poly)
    subsets = classify_vertices(contours_poly, diagonal1, diagonal2)
    if subsets is None:
        print('subset failed')
        return None
    line_params = fit_lines_to_subsets(subsets, weighted)
    try:
        intersection = get_result_intersections(line_params)
    except:
        print('no intersection')
        return None
    contour_length = contours_poly.length
    conformation = get_conformation(intersection, contour_length, normalized)

    return {'conformation': conformation, 'intersections': intersection, 'line_params': line_params}


def name_split(image_name):
    name = image_name.lower()
    pattern = re.compile(r'(\d{8})([A-Za-z]+)(vorher|nachher)(vl|vr|hl|hr)(lateral|medial)\.jpg')
    match = pattern.match(name)
    if match:
        date = match.group(1)
        horse_name = match.group(2)
        condition = match.group(3)  # Vorher or Nachher
        hoof = match.group(4)
        angle = match.group(5)
        date = datetime.strptime(date, '%Y%m%d')
        return {
            'date': date,
            'horse_name': horse_name,
            'condition': condition,
            'hoof': hoof,
            'angle': angle
        }
    else:
        return None


def creat_dataframe(image_name, conformation, inference_time):
    image_name_spilied = name_split(image_name)
    if image_name_spilied is None:
        return None
    date = image_name_spilied['date']
    date = datetime.strftime(date, '%Y-%m-%d')
    horse = image_name_spilied['horse_name']
    condition = image_name_spilied['condition']
    hoof = image_name_spilied['hoof']
    angle = image_name_spilied['angle']

    conformation_keys = conformation.keys()

    data = {'horse_name': [horse], 'hoof': [hoof], 'angle': [angle], 'condition': [condition], 'date': [date],
            'file_name': [image_name], 'inference_time': [inference_time]}
    for key in conformation_keys:
        data[key] = [conformation[key]]

    df = pd.DataFrame(data)
    return df
