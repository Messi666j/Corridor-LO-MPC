"""
栅格地图与障碍物构建工具
"""

import numpy as np
import copy


def build_grid_map(rows, cols, obstacle_regions):
    """由 (y1, y2, x1, x2) 列表构建栅格地图"""
    grid = np.zeros((rows, cols))
    for (y1, y2, x1, x2) in obstacle_regions:
        grid[y1:y2, x1:x2] = 50
    return grid


def get_inner_distance(pts):
    """计算点集内部两两之间的距离"""
    pts = np.array(pts)
    dists = []
    for i in range(len(pts) - 1):
        for j in range(i + 1, len(pts)):
            dists.append(float(np.linalg.norm(pts[i] - pts[j])))
    return dists


def make_obstacles(obstacle_regions, avoid_radius):
    """生成原始障碍物与膨胀障碍物多边形"""
    obstacles_orig, obstacles_avoid = [], []
    for (y1, y2, x1, x2) in obstacle_regions:
        obs = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
        obstacles_orig.append(copy.deepcopy(obs))
        r = avoid_radius
        obstacles_avoid.append([
            [x1 - r, y1 - r], [x1 - r, y2 + r],
            [x2 + r, y2 + r], [x2 + r, y1 - r],
        ])
    return obstacles_orig, obstacles_avoid
