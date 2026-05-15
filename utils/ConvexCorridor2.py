"""
凸安全走廊生成与 Minimum-Snap 轨迹优化模块
==============================================
从 CombinedSimulation.py 提炼出的独立可导入函数。
提供:
  - A* 栅格路径搜索
  - 路径简化 (基于转角检测)
  - 凸安全走廊生成 (矩形膨胀 + 法向切片裁剪)
  - 多边形 → 线性不等式转换
  - Minimum-Snap 轨迹优化 (CVXPY)

依赖: numpy, math, heapq, shapely, cvxpy
"""

import math
import heapq
import numpy as np
from shapely.geometry import Polygon, LineString
from shapely import vectorized
from shapely.ops import split

# ===========================================================================
#  A* 栅格路径搜索
# ===========================================================================

def astar(start, goal, grid):
    """
    在二维栅格地图上执行 A* 搜索。

    参数:
        start : (col, row) — 起点栅格坐标
        goal  : (col, row) — 终点栅格坐标
        grid  : ndarray   — 2D 栅格 (0=可通行, 其他=障碍)

    返回:
        list[(col, row)] 或 None
    """
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    def heuristic(a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    visited = set()
    while open_set:
        _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(-1,-1),(1,-1)]:
            neighbor = (current[0]+dx, current[1]+dy)
            if (0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows
                    and grid[neighbor[1], neighbor[0]] == 0):
                tentative_g = g_score[current] + math.sqrt(dx**2 + dy**2)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))
    return None


# ===========================================================================
#  路径简化
# ===========================================================================

def compute_angles(points):
    """计算路径上相邻线段夹角 (弧度)。首尾点返回 π。"""
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n < 3:
        return np.array([np.pi] * n, dtype=float)
    angles = np.empty(n, dtype=float)
    angles[0] = np.pi
    angles[-1] = np.pi
    for i in range(1, n - 1):
        v1 = points[i]   - points[i - 1]
        v2 = points[i+1] - points[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            angles[i] = 0.0
            continue
        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angles[i] = np.arccos(cos_angle)
    return angles


def simplify_path(points, corner_deg=30, corner_dilate=1):
    """
    基于转角阈值简化路径，保留关键拐点。

    参数:
        points        : array-like (n, 2)
        corner_deg    : 转角阈值 (度)
        corner_dilate : 拐点两侧扩展点数

    返回:
        ndarray (m, 2)
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n <= 2:
        return points.copy()
    angles = compute_angles(points)
    theta = np.deg2rad(corner_deg)
    corner_idx = np.where(angles >= theta)[0]
    corner_idx = np.unique(np.concatenate(([0], corner_idx, [n-1])))
    if corner_dilate > 0:
        extra = []
        for idx in corner_idx:
            for j in range(idx - corner_dilate, idx + corner_dilate + 1):
                if 0 <= j < n:
                    extra.append(j)
        corner_idx = np.unique(np.concatenate(
            (corner_idx, np.array(extra, dtype=int))))
    return points[corner_idx]


# ===========================================================================
#  法向切割线
# ===========================================================================

def create_normal_line(point1, point2, length=50):
    """
    以 point2 为中心，生成一条垂直于 (point1 -> point2) 的切割线。

    参数:
        point1 : array-like — 参考点 (如路径段中点)
        point2 : array-like — 障碍点位置
        length : float      — 切割线总长度

    返回:
        LineString
    """
    dx, dy = np.array(point2) - np.array(point1)
    vector = np.array([-dy, dx])
    norm = np.linalg.norm(vector)
    dir_vec = vector / norm if norm > 1e-6 else vector
    start = point2 + dir_vec * length / 2
    end   = point2 - dir_vec * length / 2
    return LineString([(start[0], start[1]), (end[0], end[1])])


# ===========================================================================
#  初始凸安全走廊 (矩形膨胀)
# ===========================================================================

def convex_corridor(path, grid, max_width=10.0, extend=8.0):
    """
    为每段路径生成初始矩形走廊 (沿路径方向包络)。

    参数:
        path      : array-like — 关键路径点
        grid      : ndarray    — 栅格地图
        max_width : float      — 走廊半宽度
        extend    : float      — 沿路径方向延伸量

    返回:
        list[ndarray] — 每段走廊的 4 角坐标 (x, y)
    """
    rows, cols = grid.shape
    rectangles = []
    path_xy = [(float(p[0]), float(p[1])) for p in path]

    for i in range(len(path_xy) - 1):
        p1 = np.array(path_xy[i], dtype=float)
        p2 = np.array(path_xy[i+1], dtype=float)
        seg_vec = p2 - p1
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1e-6:
            half = max_width * 0.5
            corners = np.array([
                [p1[0]-half, p1[1]-half],
                [p1[0]+half, p1[1]-half],
                [p1[0]+half, p1[1]+half],
                [p1[0]-half, p1[1]+half],
            ])
        else:
            unit = seg_vec / seg_len
            orth = np.array([-unit[1], unit[0]])
            half_w = max_width
            c1 = p1 + orth * half_w - unit * extend
            c2 = p1 - orth * half_w - unit * extend
            c3 = p2 - orth * half_w + unit * extend
            c4 = p2 + orth * half_w + unit * extend
            corners = np.vstack((c1, c2, c3, c4))
        corners[:, 0] = np.clip(corners[:, 0], 0, cols - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, rows - 1)
        rectangles.append(corners)

    return rectangles


# ===========================================================================
#  快速多边形掩膜
# ===========================================================================

def points_inside_polygon_mask_fast(poly, rows, cols):
    """判断栅格点是否在多边形内部 (Shapely 矢量化)。"""
    if len(poly) < 3:
        return np.zeros((rows, cols), dtype=bool)
    xs = np.arange(cols) + 0.5
    ys = np.arange(rows) + 0.5
    XX, YY = np.meshgrid(xs, ys)
    return vectorized.contains(Polygon(poly), XX, YY)


# ===========================================================================
#  走廊裁剪优化 (法向切片)
# ===========================================================================

def corridor_generator_optimized(path, corridor, grid, max_width=10.0):
    """
    迭代裁剪走廊多边形，切除内部障碍区域，保证走廊内无障碍。

    算法:
      1. 检测每段走廊内部障碍栅格
      2. 从路径中点向障碍点方向生成法线切割线
      3. 保留包含路径中点的子多边形
      4. 重复直至走廊内无障碍

    参数:
        path      : array-like — 简化路径点
        corridor  : list       — 初始走廊多边形 (原地修改)
        grid      : ndarray    — 栅格地图 (非 0 视为障碍)
        max_width : float      — 搜索余量

    返回:
        list — 裁剪后的走廊多边形列表
    """
    path_xy = [(float(p[0]), float(p[1])) for p in path]
    rows, cols = grid.shape

    for i in range(len(corridor)):
        poly = np.array(corridor[i], dtype=float)
        if poly.size == 0:
            continue

        min_x = int(max(0, np.floor(np.min(poly[:, 0])) - int(max_width)))
        max_x = int(min(cols - 1, np.ceil(np.max(poly[:, 0])) + int(max_width)))
        min_y = int(max(0, np.floor(np.min(poly[:, 1])) - int(max_width)))
        max_y = int(min(rows - 1, np.ceil(np.max(poly[:, 1])) + int(max_width)))

        sub_grid = grid[min_y:max_y+1, min_x:max_x+1]
        delta_row, delta_col = sub_grid.shape

        poly_local = poly.copy()
        poly_local[:, 0] -= min_x
        poly_local[:, 1] -= min_y

        mask = points_inside_polygon_mask_fast(poly_local, delta_row, delta_col)
        obs_cells = np.argwhere(mask & (sub_grid != 0))
        if obs_cells.size == 0:
            continue
        obs_cells = obs_cells + np.array([min_y, min_x])
        del_array = []

        while obs_cells.size > 0:
            r, c = obs_cells[0]
            point1 = ((path_xy[i][0] + path_xy[i+1][0]) / 2.0,
                      (path_xy[i][1] + path_xy[i+1][1]) / 2.0)
            point2 = (float(c), float(r))
            line = create_normal_line(point1, point2, length=50)
            del_array.append((r, c))

            poly_full = Polygon(corridor[i])
            result = split(poly_full, line)

            for m in range(len(result.geoms)):
                coords = np.array(result.geoms[m].exterior.coords[:-1])
                coords_local = coords.copy()
                coords_local[:, 0] -= min_x
                coords_local[:, 1] -= min_y
                mask1 = points_inside_polygon_mask_fast(
                    coords_local, delta_row, delta_col)

                mid_x = (path_xy[i][0] + path_xy[i+1][0]) / 2.0 - min_x
                mid_y = (path_xy[i][1] + path_xy[i+1][1]) / 2.0 - min_y

                if not (0 <= int(np.floor(mid_y)) < delta_row
                        and 0 <= int(np.floor(mid_x)) < delta_col):
                    continue

                if mask1[int(np.floor(mid_y)), int(np.floor(mid_x))]:
                    corridor[i] = coords
                    mask = mask1
                    obs_cells_local = np.argwhere(mask & (sub_grid != 0))
                    obs_cells = obs_cells_local + np.array([min_y, min_x])
                    if del_array:
                        obs_cells = np.array([
                            row for row in obs_cells
                            if not any((row == np.array(del_array)).all(axis=1))
                        ])
                    break

    return corridor


# ===========================================================================
#  多边形 → 线性不等式
# ===========================================================================

def polygon_to_inequalities(vertices):
    """
    将凸多边形顶点转换为 Ax <= b 形式。

    参数:
        vertices : ndarray (m, 2) — 多边形顶点坐标

    返回:
        A : ndarray (m, 2), b : ndarray (m,)
    """
    n = len(vertices)
    A, b_list = [], []
    center = np.mean(vertices, axis=0)
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        # 确保法向量指向多边形外部
        if np.dot(normal, center - p1) > 0:
            normal = -normal
        A.append(normal)
        b_list.append(np.dot(normal, p1))
    return np.array(A), np.array(b_list)


# ===========================================================================
#  Minimum-Snap 轨迹优化 (CVXPY)
# ===========================================================================

def _make_diff_matrix(order, N):
    """构造 order 阶有限差分矩阵 (N 个轨迹点)。"""
    if order >= N:
        return np.zeros((0, N))
    S = np.zeros((N - order, N))
    coeff = np.array([1.])
    for _ in range(order):
        coeff = np.convolve(coeff, np.array([1, -1]))
    for i in range(N - order):
        S[i, i:i+order+1] = coeff
    return S


def _compute_centroid(polygon):
    """计算多边形质心坐标。"""
    n = len(polygon)
    if n < 3:
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        return (sum(xs) / n, sum(ys) / n)
    area = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    area *= 0.5
    if abs(area) < 1e-12:
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        return (sum(xs) / n, sum(ys) / n)
    cx /= (6 * area)
    cy /= (6 * area)
    return (cx, cy)


def minimum_snap_solver(corridor, grid, path, N, dim, solver="OSQP",
                        lambda_center=5.0):
    """
    求解 Minimum-Snap 轨迹优化问题，输出平滑轨迹。

    参数:
        corridor      : list — 凸多边形走廊 (每个为 ndarray (m,2))
        grid          : ndarray — 栅格地图
        path          : array-like — 简化路径点
        N             : int — 轨迹点数
        dim           : int — 维度 (通常 2)
        solver        : str — "OSQP", "ECOS", "SCS"
        lambda_center : float — 引导轨迹靠近走廊中心的权重

    返回:
        ndarray (N, dim) — 优化后轨迹点
    """
    import cvxpy as cp

    # 选择导数阶数
    if N >= 5:
        S = _make_diff_matrix(4, N)
        obj_expr = lambda X: cp.sum_squares(S @ X[:, 0]) + \
                              cp.sum_squares(S @ X[:, 1])
    elif N >= 3:
        S = _make_diff_matrix(2, N)
        obj_expr = lambda X: cp.sum_squares(S @ X[:, 0]) + \
                              cp.sum_squares(S @ X[:, 1])
    else:
        S = _make_diff_matrix(1, N)
        obj_expr = lambda X: cp.sum_squares(S @ X[:, 0]) + \
                              cp.sum_squares(S @ X[:, 1])

    traj = cp.Variable((N, dim))
    constraints = []
    constraints += [traj[0, :] == path[0]]
    constraints += [traj[-1, :] == path[-1]]

    # 走廊约束
    for i in range(N - 1):
        poly = corridor[i]
        if poly is None or len(poly) < 3:
            constraints += [traj[i, 0] >= 0,
                            traj[i, 0] <= grid.shape[1] - 1]
            constraints += [traj[i, 1] >= 0,
                            traj[i, 1] <= grid.shape[0] - 1]
        else:
            A_poly, b_poly = polygon_to_inequalities(np.array(poly))
            for k in range(A_poly.shape[0]):
                arow = A_poly[k]
                brow = b_poly[k] + 1e-6
                constraints += [
                    arow[0] * traj[i, 0] + arow[1] * traj[i, 1] <= brow]

    # 最后一个点约束
    if len(corridor) >= 1:
        A_poly, b_poly = polygon_to_inequalities(np.array(corridor[-1]))
        for k in range(A_poly.shape[0]):
            arow = A_poly[k]
            brow = b_poly[k] + 1e-6
            constraints += [
                arow[0] * traj[-1, 0] + arow[1] * traj[-1, 1] <= brow]

    # 质心引导项
    centers = []
    for i in range(N - 1):
        poly = corridor[i]
        if len(poly) < 3:
            cx = (path[i][0] + path[i+1][0]) / 2
            cy = (path[i][1] + path[i+1][1]) / 2
            centers.append((cx, cy))
        else:
            centers.append(_compute_centroid(poly))
    poly = corridor[-1]
    if len(poly) < 3:
        cx = (path[-2][0] + path[-1][0]) / 2
        cy = (path[-2][1] + path[-1][1]) / 2
        centers.append((cx, cy))
    else:
        centers.append(_compute_centroid(poly))
    centers = np.array(centers)

    centers_param = cp.Parameter(shape=(N, 2))
    centers_param.value = centers
    center_term = cp.sum(cp.sum_squares(traj - centers_param))
    objective = cp.Minimize(obj_expr(traj) + lambda_center * center_term)

    prob = cp.Problem(objective, constraints)
    try:
        if solver == "ECOS":
            prob.solve(solver=cp.ECOS, verbose=False)
        elif solver == "OSQP":
            prob.solve(solver=cp.OSQP, verbose=False,
                       eps_abs=1e-3, eps_rel=1e-3, max_iter=20000)
        elif solver == "SCS":
            prob.solve(solver=cp.SCS, verbose=False)
        else:
            raise NameError(f"Unknown solver: {solver}")
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)

    if traj.value is None:
        return np.array([(float(p[1]), float(p[0])) for p in path])
    return traj.value
