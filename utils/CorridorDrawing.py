"""
安全走廊与编队可视化绘图函数

包含安全走廊可视化、路径与编队状态绘制、性能指标图表生成。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle
from utils.Drawing import compute_vertices


def draw_corridor_only(ax, corridor, obstacles_avoid, obstacles_orig, grid, ROWS, COLS,
                       formation_radius, start_pt, goal_pt):
    """仅绘制膨胀的障碍物与安全走廊"""
    ax.imshow(grid, origin="lower", cmap="gray_r", alpha=0.6,
              extent=[0, COLS, 0, ROWS])

    for rect in corridor:
        if rect is not None and len(rect) >= 3:
            ax.add_patch(MplPolygon(rect, closed=True, alpha=0.3,
                                    facecolor='orange', edgecolor='red',
                                    linewidth=1.5))

    for obs_avoid in obstacles_avoid:
        arr = np.array(obs_avoid)
        rect = plt.Rectangle(arr[0], arr[2][0] - arr[0][0], arr[2][1] - arr[0][1],
                             facecolor='pink', alpha=0.25, edgecolor='pink',
                             linestyle=':', linewidth=1.0, zorder=2)
        ax.add_patch(rect)

    for obs in obstacles_orig:
        arr = np.array(obs)
        rect = plt.Rectangle(arr[0], arr[2][0] - arr[0][0], arr[2][1] - arr[0][1],
                             facecolor='black', edgecolor='black',
                             alpha=1.0, zorder=1)
        ax.add_patch(rect)

    formation_circle = Circle((start_pt[0], start_pt[1]), formation_radius,
                              color='blue', fill=False, linestyle='-.',
                              linewidth=1.5, zorder=4, label='Formation Boundary')
    ax.add_patch(formation_circle)

    ax.plot(start_pt[0], start_pt[1], 'go', markersize=12, label='Start',
            markeredgecolor='darkgreen', markeredgewidth=2, zorder=5)
    ax.plot(goal_pt[0], goal_pt[1], 'r*', markersize=15,
            label='Goal', markeredgecolor='darkred', zorder=5)

    ax.set_xlim(-1, COLS + 1)
    ax.set_ylim(-1, ROWS + 1)
    ax.set_xlabel('X (columns)', fontsize=14)
    ax.set_ylabel('Y (rows)', fontsize=14)
    ax.set_title('Inflated Obstacles & Safety Corridors', fontsize=16, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')


def draw_paths_and_stages(ax, obstacles_orig, path_simplified, traj_opt, raw_path,
                          ROWS, COLS, start_pt, goal_pt, cargo_center, formation_radius,
                          stage_snapshots, traj_data, colors):
    """显示障碍物、路径和各stage编队状态叠加"""
    # 网格背景
    grid = np.zeros((ROWS, COLS))
    ax.imshow(grid, origin="lower", cmap="gray_r", alpha=0.6,
              extent=[0, COLS, 0, ROWS])

    # 原始障碍物
    for obs in obstacles_orig:
        arr = np.array(obs)
        rect = plt.Rectangle(arr[0], arr[2][0] - arr[0][0], arr[2][1] - arr[0][1],
                             color='black', zorder=3)
        ax.add_patch(rect)

    # 简化路径
    if len(path_simplified) > 1:
        ax.plot(path_simplified[:, 0], path_simplified[:, 1], 'ro--',
                linewidth=1, markersize=6, label='Simplified path',
                markerfacecolor='red', markeredgecolor='darkred')

    # MinSnap轨迹
    if len(traj_opt) > 1:
        ax.plot(traj_opt[:, 0], traj_opt[:, 1], 'bo-',
                linewidth=2, markersize=6, label='Minimumsnap path',
                markerfacecolor='blue', markeredgecolor='darkblue')

    # 原始A*路径
    if len(raw_path) > 1:
        path_arr = np.array(raw_path)
        ax.plot(path_arr[:, 0], path_arr[:, 1], 'y--',
                alpha=0.3, linewidth=1, label='Original A* path')

    # 编队边界圆
    formation_circle = Circle((cargo_center[0], cargo_center[1]), formation_radius,
                              color='blue', fill=False, linestyle='-.',
                              linewidth=1.5, zorder=4, label='Formation Boundary')
    ax.add_patch(formation_circle)

    # 起点/终点
    ax.plot(start_pt[0], start_pt[1], 'go', markersize=12, label='Start',
            markeredgecolor='darkgreen', markeredgewidth=2, zorder=5)
    ax.plot(goal_pt[0], goal_pt[1], 'r*', markersize=15,
            label='Goal', markeredgecolor='darkred', zorder=5)

    # 各stage快照（透明度递增）
    n_stages = len(stage_snapshots)
    for i, snapshot in enumerate(stage_snapshots):
        alpha = 0.20 + 0.55 * (i / max(n_stages - 1, 1))
        for j, (xr, yr, th) in enumerate(snapshot):
            verts = compute_vertices(4.0, 3.0, xr, yr, th)
            ax.plot(verts[0], verts[1], color=colors[j], alpha=alpha,
                    linewidth=1.2, zorder=4)
        cargo_x = [s[0] for s in snapshot] + [snapshot[0][0]]
        cargo_y = [s[1] for s in snapshot] + [snapshot[0][1]]
        ax.plot(cargo_x, cargo_y, 'k-', alpha=alpha * 0.6, linewidth=0.8, zorder=3)

    # 完整轨迹（虚线）
    for j in range(len(colors)):
        if 2 * j + 1 < len(traj_data):
            ax.plot(traj_data[2 * j], traj_data[2 * j + 1], color=colors[j],
                    linestyle='--', linewidth=1, alpha=0.5, zorder=3)

    ax.set_xlim(-1, COLS + 1)
    ax.set_ylim(-1, ROWS + 1)
    ax.set_xlabel('X (columns)', fontsize=14)
    ax.set_ylabel('Y (rows)', fontsize=14)
    ax.set_title('Obstacles, Planned Paths & Formation Stages', fontsize=16, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')


def draw_all_stages_combined(ax, all_stages_data, obstacles_orig,
                             ROWS, COLS, start_pt, goal_pt, traj_data, colors):
    """所有阶段编队状态组合图"""
    ax.set_facecolor('white')

    # 原始障碍物
    for obs in obstacles_orig:
        arr = np.array(obs)
        rect = plt.Rectangle(arr[0], arr[2][0] - arr[0][0], arr[2][1] - arr[0][1],
                             facecolor='black', edgecolor='black',
                             alpha=1.0, zorder=2)
        ax.add_patch(rect)

    # 各AGV完整轨迹（虚线）
    for j in range(len(colors)):
        if 2 * j + 1 < len(traj_data):
            ax.plot(traj_data[2 * j], traj_data[2 * j + 1],
                    color=colors[j], linestyle='--', linewidth=1.2,
                    alpha=0.55, zorder=3)

    # 质心轨迹
    if traj_data:
        cx = np.mean([traj_data[2*j] for j in range(len(colors)) if 2*j < len(traj_data)], axis=0)
        cy = np.mean([traj_data[2*j+1] for j in range(len(colors)) if 2*j+1 < len(traj_data)], axis=0)
        ax.plot(cx, cy, 'k:', linewidth=1.2, alpha=0.45, zorder=3, label='Centroid path')

    # 各stage编队轮廓（统一透明度）
    for sd in all_stages_data:
        if 'cargo_poly_x' in sd and 'cargo_poly_y' in sd:
            ax.plot(sd['cargo_poly_x'], sd['cargo_poly_y'], 'k-',
                    alpha=0.60, linewidth=1.2, zorder=4)

        for j, (x, y, th) in enumerate(sd['robots']):
            verts = compute_vertices(4.0, 3.0, x, y, th)
            ax.plot(verts[0], verts[1], color=colors[j], alpha=0.75,
                    linewidth=1.8, zorder=4)

    # 起点/终点
    ax.plot(start_pt[0], start_pt[1], 'go', markersize=12,
            markeredgecolor='darkgreen', markeredgewidth=2, zorder=5, label='Start')
    ax.plot(goal_pt[0], goal_pt[1], 'r*', markersize=15,
            markeredgecolor='darkred', zorder=5, label='Goal')

    ax.set_xlim(-1, COLS + 1)
    ax.set_ylim(-1, ROWS + 1)
    ax.set_xlabel('X (columns)', fontsize=14)
    ax.set_ylabel('Y (rows)', fontsize=14)
    ax.set_title('All Formation Stages Combined', fontsize=16, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.15, linestyle='--')


def draw_trajectory_overview(ax, corridor, obstacles_avoid, obstacles_orig,
                             grid, ROWS, COLS, formation_radius,
                             path_simplified, traj_opt, raw_path,
                             start_pt, goal_pt, cargo_center):
    """膨胀障碍物 + 安全走廊 + 规划路径总览（无机器人编队轮廓）"""
    # 网格背景
    ax.imshow(grid, origin="lower", cmap="gray_r", alpha=0.4,
              extent=[0, COLS, 0, ROWS])

    # 安全走廊多边形
    for rect in corridor:
        if rect is not None and len(rect) >= 3:
            ax.add_patch(MplPolygon(rect, closed=True, alpha=0.25,
                                    facecolor='orange', edgecolor='red',
                                    linewidth=1.5, zorder=2))

    # 膨胀障碍物（粉色安全边界）
    for obs_avoid in obstacles_avoid:
        arr = np.array(obs_avoid)
        rect = plt.Rectangle(arr[0], arr[2][0] - arr[0][0], arr[2][1] - arr[0][1],
                             facecolor='pink', alpha=0.30, edgecolor='pink',
                             linestyle=':', linewidth=1.0, zorder=2)
        ax.add_patch(rect)

    # 原始障碍物（黑色实心）
    for obs in obstacles_orig:
        arr = np.array(obs)
        rect = plt.Rectangle(arr[0], arr[2][0] - arr[0][0], arr[2][1] - arr[0][1],
                             facecolor='black', edgecolor='black',
                             alpha=1.0, zorder=3)
        ax.add_patch(rect)

    # 原始A*路径（淡黄色虚线）
    if len(raw_path) > 1:
        path_arr = np.array(raw_path)
        ax.plot(path_arr[:, 0], path_arr[:, 1], 'y--',
                alpha=0.25, linewidth=1, label='Original A* path')

    # 简化路径（红色）
    if len(path_simplified) > 1:
        ax.plot(path_simplified[:, 0], path_simplified[:, 1], 'ro--',
                linewidth=1.5, markersize=7, label='Simplified path',
                markerfacecolor='red', markeredgecolor='darkred')

    # MinSnap轨迹（蓝色）
    if len(traj_opt) > 1:
        ax.plot(traj_opt[:, 0], traj_opt[:, 1], 'bo-',
                linewidth=2, markersize=7, label='Minimumsnap path',
                markerfacecolor='blue', markeredgecolor='darkblue')

    # 编队边界圆
    formation_circle = Circle((cargo_center[0], cargo_center[1]), formation_radius,
                              color='blue', fill=False, linestyle='-.',
                              linewidth=1.5, zorder=4, label='Formation Boundary')
    ax.add_patch(formation_circle)

    # 起点/终点
    ax.plot(start_pt[0], start_pt[1], 'go', markersize=12, label='Start',
            markeredgecolor='darkgreen', markeredgewidth=2, zorder=5)
    ax.plot(goal_pt[0], goal_pt[1], 'r*', markersize=15,
            label='Goal', markeredgecolor='darkred', zorder=5)

    ax.set_xlim(-1, COLS + 1)
    ax.set_ylim(-1, ROWS + 1)
    ax.set_xlabel('X (columns)', fontsize=14)
    ax.set_ylabel('Y (rows)', fontsize=14)
    ax.set_title('Safe Corridors & Planned Trajectories', fontsize=16, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.2, linestyle='--')


def draw_performance_plots(save_dir, time_hist, hist_v, hist_w, v_max_limit, w_max_limit,
                           hist_err, pair_labels, ROBOT_NUM, vehicle_colors):
    """绘制并保存性能指标图（速度、角速度、编队误差）"""
    t = np.array(time_hist)

    # ---------- 图 1: 各机器人线速度 ----------
    fig_v, ax_v = plt.subplots(figsize=(7, 4))
    for j in range(ROBOT_NUM):
        ax_v.plot(t, hist_v[j], color=vehicle_colors[j], linewidth=1.2,
                  label=f'AGV {j + 1}')
    ax_v.axhline(y=v_max_limit, color='red', linestyle='--', linewidth=0.8,
                 alpha=0.7, label=f'$v_{{\\mathrm{{max}}}}$={v_max_limit}')
    ax_v.axhline(y=-v_max_limit, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
    ax_v.set_xlabel('t (s)', fontsize=12)
    ax_v.set_ylabel('Linear Velocity (m/s)', fontsize=12)
    ax_v.set_title('Linear Velocity of Each AGV', fontsize=13, fontweight='bold')
    ax_v.legend(fontsize=9, loc='best')
    ax_v.grid(True, alpha=0.3, linestyle='--')
    fig_v.tight_layout()
    fig_v.savefig(os.path.join(save_dir, 'linear_velocity.png'), dpi=150, bbox_inches='tight')
    print("  -> Saved linear_velocity.png")
    plt.close(fig_v)

    # ---------- 图 2: 各机器人角速度 ----------
    fig_w, ax_w = plt.subplots(figsize=(7, 4))
    for j in range(ROBOT_NUM):
        ax_w.plot(t, hist_w[j], color=vehicle_colors[j], linewidth=1.2,
                  label=f'AGV {j + 1}')
    ax_w.axhline(y=w_max_limit, color='red', linestyle='--', linewidth=0.8,
                 alpha=0.7, label=f'$\\omega_{{\\mathrm{{max}}}}$={w_max_limit}')
    ax_w.axhline(y=-w_max_limit, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
    ax_w.set_xlabel('t (s)', fontsize=12)
    ax_w.set_ylabel('Angular Velocity (rad/s)', fontsize=12)
    ax_w.set_title('Angular Velocity of Each AGV', fontsize=13, fontweight='bold')
    ax_w.legend(fontsize=9, loc='best')
    ax_w.grid(True, alpha=0.3, linestyle='--')
    fig_w.tight_layout()
    fig_w.savefig(os.path.join(save_dir, 'angular_velocity.png'), dpi=150, bbox_inches='tight')
    print("  -> Saved angular_velocity.png")
    plt.close(fig_w)

    # ---------- 图 3: 角速度差异分析 ----------
    fig_diff, ax_diff = plt.subplots(figsize=(10, 6))
    for i in range(ROBOT_NUM):
        for j in range(i + 1, ROBOT_NUM):
            omega_diff = np.array(hist_w[i]) - np.array(hist_w[j])
            ax_diff.plot(t, omega_diff, linewidth=1.2,
                         label=f'$\\omega_{{{i+1}}} - \\omega_{{{j+1}}}$')
    ax_diff.set_xlabel('t (s)', fontsize=12)
    ax_diff.set_ylabel('Angular Velocity Difference (rad/s)', fontsize=12)
    ax_diff.set_title('Inter-AGV Angular Velocity Differences During Turning', fontsize=13, fontweight='bold')
    ax_diff.legend(fontsize=9, loc='best')
    ax_diff.grid(True, alpha=0.3, linestyle='--')
    fig_diff.tight_layout()
    fig_diff.savefig(os.path.join(save_dir, 'angular_velocity_difference.png'), dpi=150, bbox_inches='tight')
    print("  -> Saved angular_velocity_difference.png")
    plt.close(fig_diff)

    # ---------- 图 4: 编队内部距离误差 ----------
    num_pairs = len(pair_labels)
    fig_e, ax_e = plt.subplots(figsize=(7, 4))
    err_colors = ['gray', 'orange', 'green', 'steelblue', 'purple', 'brown']
    for k in range(num_pairs):
        ax_e.plot(t, hist_err[k], color=err_colors[k % len(err_colors)],
                  linewidth=1.2, label=pair_labels[k])
    ax_e.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax_e.set_xlabel('t (s)', fontsize=12)
    ax_e.set_ylabel('Formation Error (m)', fontsize=12)
    ax_e.set_title('Inter-Robot Distance Errors', fontsize=13, fontweight='bold')
    ax_e.legend(fontsize=9, loc='best', ncol=2)
    ax_e.grid(True, alpha=0.3, linestyle='--')
    fig_e.tight_layout()
    fig_e.savefig(os.path.join(save_dir, 'formation_error.png'), dpi=150, bbox_inches='tight')
    print("  -> Saved formation_error.png")
    plt.close(fig_e)
