"""
LO-MPC 编队导航 + 凸安全走廊避障仿真
=====================================
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
import time

import casadi as cs
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle
from scipy.ndimage import binary_dilation

from lexHMPC.HMPC import (
    extract_input, solve_nlp, add_lex_constraint,
    cal_xc, compute_obs_coeff, compute_obs_params
)
from lexHMPC import Form, Navi, Delta_U, HMPC, Orient
from utils.Drawing import compute_vertices
from utils.ConvexCorridor2 import (
    astar, convex_corridor, corridor_generator_optimized,
    simplify_path, minimum_snap_solver
)
from utils.MapBuilder import build_grid_map, get_inner_distance, make_obstacles
from utils.CorridorDrawing import (
    draw_corridor_only, draw_paths_and_stages,
    draw_all_stages_combined, draw_trajectory_overview,
    draw_performance_plots
)

# ===========================================================================
# 仿真元参数
# ===========================================================================

SAVE = True
MAX_ITER = 3000

# ===========================================================================
# Part 1: 构建栅格地图
# ===========================================================================

ROWS, COLS = 100, 150

# 活动地图配置 — 修改此列表即可自定义障碍物布局
# 格式: (y_min, y_max, x_min, x_max)
obstacle_regions1 = [
    (30, 45, 25, 35),
    (70, 85, 45, 50),
    (15, 37, 65, 70),
    (60, 75, 85, 95),
    (30, 40, 115, 125),
    (20, 40, 135, 150),
]

#备选地图
obstacle_regions2 = [
    (30, 45, 25, 35),
    (70, 85, 45, 50),
    (5, 27, 70, 75),
    (60, 75, 85, 95),
    (30, 40, 115, 125),
    (20, 40, 135, 150),
    (45, 50, 100, 110),
]
obstacle_regions3 = [
    (30, 45, 25, 35),
    (70, 85, 45, 50),
    (5, 27, 70, 75),
    (60, 75, 85, 95),
    (30, 40, 115, 125),
    (20, 40, 135, 150),
    (60, 65, 65, 75),
]

obstacle_regions = obstacle_regions2 # 修改这里以选择地图，或自定义地图
grid = build_grid_map(ROWS, COLS, obstacle_regions)

# ===========================================================================
# Part 2: 编队与障碍物参数
# ===========================================================================

ROBOT_NUM = 4
LENGTH, WIDTH = 4.0, 3.0

cargo_half = 4.0
cargo_cx, cargo_cy = 15.0, 15.0
cargo = np.array([
    [cargo_cx - cargo_half, cargo_cy - cargo_half],
    [cargo_cx - cargo_half, cargo_cy + cargo_half],
    [cargo_cx + cargo_half, cargo_cy + cargo_half],
    [cargo_cx + cargo_half, cargo_cy - cargo_half],
])
cargo_center = [float(np.mean(cargo[:, 0])), float(np.mean(cargo[:, 1]))]
cargo_radius = float(np.max(np.linalg.norm(cargo - cargo_center, axis=1)))
formation_radius = cargo_radius + math.sqrt((LENGTH / 2)**2 + (WIDTH / 2)**2)

obstacles_orig, obstacles_avoid = make_obstacles(obstacle_regions, formation_radius)

df_star = get_inner_distance(cargo)
destination = np.array([140.0, 50.0])

# ===========================================================================
# Part 3: A* + 凸安全走廊
# ===========================================================================

print("=" * 60)
print("  LO-MPC + Convex Safe Corridor Formation Navigation")
print("=" * 60)

# 膨胀栅格
inflate_r = int(math.ceil(formation_radius))
print(f"  Inflate radius: {inflate_r} (formation_radius = {formation_radius:.1f})")
obs_mask = (grid == 50)
struct = np.ones((2 * inflate_r + 1, 2 * inflate_r + 1))
grid_safe = np.where(binary_dilation(obs_mask, structure=struct), 50, 0).astype(float)

# A* 搜索
start_pt = (int(cargo_cx), int(cargo_cy))
goal_pt = (int(destination[0]), int(destination[1]))
raw_path = astar(start_pt, goal_pt, grid_safe)
if raw_path is None:
    raise RuntimeError("A* failed to find a path!")
print(f"  A* path: {len(raw_path)} pts")

# 路径简化
path_simplified = simplify_path(np.array(raw_path, dtype=float),
                                corner_deg=30, corner_dilate=1)
print(f"  Simplified: {len(path_simplified)} keypoints")

# 凸安全走廊
max_width = 10.0
corridor = convex_corridor(path_simplified, grid_safe, max_width=max_width, extend=8)
corridor = corridor_generator_optimized(path_simplified, corridor, grid_safe, max_width)
print(f"  Corridors: {len(corridor)}")

# MinSnap 轨迹优化
N_ms = len(path_simplified)
traj_opt = minimum_snap_solver(corridor, grid_safe, raw_path, N_ms, dim=2, solver="OSQP")
navi_points = [traj_opt[i] for i in range(len(traj_opt))]
print(f"  MinSnap waypoints: {len(navi_points)}")

# 基于MinSnap轨迹计算每个waypoint的切线方向作为编队期望朝向
theta_d_list = []
for i in range(len(traj_opt) - 1):
    dx = traj_opt[i + 1][0] - traj_opt[i][0]
    dy = traj_opt[i + 1][1] - traj_opt[i][1]
    theta_d_list.append(math.atan2(dy, dx))
theta_d_list.append(theta_d_list[-1])

# ===========================================================================
# Part 4: LO-MPC 初始化
# ===========================================================================

print(f"\n[Step 4/4] Init LO-MPC...")

N, dt = 4, 0.15
a = [0, 0.5, 0.5, 0]
theta_d = theta_d_list[0]

params = {
    'robot_num': ROBOT_NUM, 'length': LENGTH, 'width': WIDTH,
    'destination': destination, 'obs_type': 'polygonal',
    'obs_num': len(obstacles_orig), 'cargo': cargo.tolist(),
    'cargo_radius': cargo_radius, 'cargo_center': [cargo_center],
    'df_star': df_star, 'N': N, 'dt': dt,
    'v_max': WIDTH, 'w_max': 1.0, 'theta_d': theta_d,
    'a': a,
    'obstacles': obstacles_orig, 'obstacles_avoid': obstacles_avoid,
    'obstacles_dyna': [], 'obstacles_dyna_center': [],
    'obstacles_dyna_radius': [],
}

obs_num, total_side_num = compute_obs_params(params)
A_list, b_list = compute_obs_coeff(params)

x_current = []
for vertex in cargo:
    x_current += [float(vertex[0]), float(vertex[1]), 0.0]
x_current = cs.SX(x_current)
u_current = cs.SX([0] * (2 * ROBOT_NUM))

x = cs.SX.sym('x', 3 * ROBOT_NUM, N)
u = cs.SX.sym('u', 2 * ROBOT_NUM, N - 1)
lam = cs.SX.sym('lam', total_side_num, N)

form_mpc = Form.Formation(params)
navi_mpc = Navi.Navigation(params)
orient_mpc = Orient.Orientation(params)
delta_u_mpc = Delta_U.Delta_U(params)

l = Navi.cal_l(params, a, cargo.tolist())
cost_form = form_mpc.obj_setup(x)
cost_navi = navi_mpc.obj_setup(x, x_current, navi_points[0], a, theta_d, l)
cost_orient = orient_mpc.obj_setup(x, a, theta_d, l)

tasks = [form_mpc, navi_mpc, orient_mpc]
costs = [cost_form, cost_navi, cost_orient]
tolerances = [0.01 * N, 0.01 * N, 0.01 * N]

hmpc = HMPC.HMPC(params, tasks)
constraints_0, lbg_0, ubg_0, lbx, ubx = hmpc.universal_constraints(x, u, lam)

x_index_end = 3 * ROBOT_NUM * N
u_current_index_end = 2 * ROBOT_NUM
last_solved = None

# ===========================================================================
# Part 5: 静态图像与动画初始化
# ===========================================================================

print(f"\n[Step 5/5] Generating static images...")

# 图像1：仅显示膨胀的障碍物与安全走廊
fig1, ax1 = plt.subplots(figsize=(14, 9))
draw_corridor_only(ax1, corridor, obstacles_avoid, obstacles_orig,
                   grid, ROWS, COLS, formation_radius, start_pt, goal_pt)
if SAVE:
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'main')
    os.makedirs(save_dir, exist_ok=True)
    fig1.savefig(os.path.join(save_dir, 'corridor_only.png'), bbox_inches='tight')
    print("  -> Saved corridor_only.png")
plt.close(fig1)

# 创建动画 figure
fig_anim, ax_anim = plt.subplots(figsize=(14, 9))
plt.ion()

# 网格背景
ax_anim.imshow(grid, origin="lower", cmap="gray_r", alpha=0.6,
               extent=[0, COLS, 0, ROWS])

# 安全走廊多边形
for rect in corridor:
    if rect is not None and len(rect) >= 3:
        ax_anim.add_patch(MplPolygon(rect, closed=True, alpha=0.3,
                                     facecolor='orange', edgecolor='red',
                                     linewidth=1.5))

# 简化路径
if len(path_simplified) > 1:
    ax_anim.plot(path_simplified[:, 0], path_simplified[:, 1], 'ro--',
                 linewidth=1, markersize=6, label='Simplified path',
                 markerfacecolor='red', markeredgecolor='darkred')

# MinSnap 轨迹
if len(traj_opt) > 1:
    ax_anim.plot(traj_opt[:, 0], traj_opt[:, 1], 'bo-',
                 linewidth=2, markersize=6, label='Minimumsnap path',
                 markerfacecolor='blue', markeredgecolor='darkblue')

# 原始 A* 路径
if len(raw_path) > 1:
    path_arr = np.array(raw_path)
    ax_anim.plot(path_arr[:, 0], path_arr[:, 1], 'y--',
                 alpha=0.3, linewidth=1, label='Original A* path')

# 膨胀障碍物（粉色安全边界）
for obs_avoid in obstacles_avoid:
    arr = np.array(obs_avoid)
    rect = plt.Rectangle(arr[0], arr[2][0] - arr[0][0], arr[2][1] - arr[0][1],
                         facecolor='pink', alpha=0.25, edgecolor='pink',
                         linestyle=':', linewidth=1.0, zorder=2)
    ax_anim.add_patch(rect)

# 原始障碍物（黑色实心）
for obs in obstacles_orig:
    arr = np.array(obs)
    rect = plt.Rectangle(arr[0], arr[2][0] - arr[0][0], arr[2][1] - arr[0][1],
                         color='black', zorder=3)
    ax_anim.add_patch(rect)

# 起点 / 终点
ax_anim.plot(start_pt[0], start_pt[1], 'go', markersize=12, label='Start',
             markeredgecolor='darkgreen', markeredgewidth=2, zorder=5)
ax_anim.plot(goal_pt[0], goal_pt[1], 'r*', markersize=15,
             label='Goal', markeredgecolor='darkred', zorder=5)

# 当前导航目标标记
navi_marker, = ax_anim.plot(navi_points[0][0], navi_points[0][1], 'ms',
                            markersize=10, label='Current target', zorder=5,
                            markeredgecolor='purple', markeredgewidth=2)

# 机器人绘图元素初始化
colors = ['tomato', 'orange', 'royalblue', 'darkgreen']
robot_drawers, traj_drawers, traj_data = [], [], []
for j in range(ROBOT_NUM):
    xr = float(x_current[3 * j])
    yr = float(x_current[3 * j + 1])
    th = float(x_current[3 * j + 2])
    verts = compute_vertices(LENGTH, WIDTH, xr, yr, th)
    rd, = ax_anim.plot(verts[0], verts[1], color=colors[j],
                       linewidth=2, label=f'AGV {j + 1}', zorder=4)
    robot_drawers.append(rd)
    traj_data.append([xr])
    traj_data.append([yr])
    td, = ax_anim.plot(xr, yr, color=colors[j], linestyle='--',
                       linewidth=1, alpha=0.6)
    traj_drawers.append(td)

# 质心轨迹
traj_c = [[cargo_cx], [cargo_cy]]
traj_c_drawer, = ax_anim.plot(traj_c[0], traj_c[1], color='darkred',
                              linewidth=1.5, label='Centroid', zorder=4)

# 编队轮廓（初始为空，在循环中更新）
cargo_drawer, = ax_anim.plot([], [], 'k-', linewidth=1.5, zorder=4)

# 编队边界圆
formation_circle = Circle((cargo_cx, cargo_cy), formation_radius,
                          color='blue', fill=False, linestyle='-.',
                          linewidth=1.5, zorder=4, label='Formation Boundary')
ax_anim.add_patch(formation_circle)

# 坐标轴设置
ax_anim.set_xlim(-1, COLS + 1)
ax_anim.set_ylim(-1, ROWS + 1)
ax_anim.set_xlabel('X (columns)', fontsize=14)
ax_anim.set_ylabel('Y (rows)', fontsize=14)
ax_anim.set_title('LO-MPC + Convex Safe Corridor Formation Navigation', fontsize=16, fontweight='bold')
ax_anim.set_aspect('equal')
ax_anim.legend(loc='upper left', fontsize=10, ncol=2)
ax_anim.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()

# ===========================================================================
# Part 6: 仿真主循环
# ===========================================================================

# 指标记录容器
v_max_limit = params['v_max']
w_max_limit = params['w_max']

hist_v = [[] for _ in range(ROBOT_NUM)]
hist_w = [[] for _ in range(ROBOT_NUM)]

pair_labels = []
for i in range(ROBOT_NUM - 1):
    for j in range(i + 1, ROBOT_NUM):
        pair_labels.append(f"$e_{{{i+1},{j+1}}}$")
num_pairs = len(pair_labels)
hist_err = [[] for _ in range(num_pairs)]
time_hist = []

# 用于记录所有stage的数据
all_stages_data = []
stage_snapshots = []

it = 0
index_navi = 0
time_start = time.time()

while it < MAX_ITER:
    # 起始约束
    constraints, lbg, ubg = HMPC.start_constraint(
        params, x, x_current, u, u_current, constraints_0, lbg_0, ubg_0)

    # 词典序优化
    for i in range(len(tasks)):
        s = solve_nlp(params, x, u, lam, costs[i],
                      constraints, lbg, ubg, lbx, ubx, last_solved)
        f_val = s['f']
        add_lex_constraint(tasks[i].obj_func, f_val, tolerances[i],
                           constraints, lbg, ubg)
    last_solved = s['x']

    # 导航点切换
    xc = cal_xc(params, x_current)
    dist_to_navi = float(cs.sumsqr(xc - navi_points[index_navi]))

    if index_navi == len(navi_points) - 1:
        if dist_to_navi < 2.0:
            err_orient = orient_mpc.cal_orient_err(x_current, a, theta_d)
            if abs(err_orient) < 10 / 180 * math.pi or dist_to_navi < 0.5:
                print(f"\n[Done] Reached destination! Iterations: {it}")
                break
    else:
        if dist_to_navi < cargo_radius * 1.5:
            index_navi += 1
            theta_d = theta_d_list[index_navi]
            cost_navi = navi_mpc.obj_setup(
                x, x_current, navi_points[index_navi], a, theta_d, l)
            costs[1] = cost_navi
            cost_orient = orient_mpc.obj_setup(x, a, theta_d, l)
            costs[2] = cost_orient
            print(f"  -> Switch waypoint {index_navi}/{len(navi_points) - 1} "
                  f"(dist={math.sqrt(dist_to_navi):.1f})")

    # 提取控制输入并更新状态
    u_current = extract_input(last_solved, x_index_end, u_current_index_end)
    HMPC.update_state(params, x_current, u_current)

    # 记录指标
    time_hist.append(it * dt)
    for j in range(ROBOT_NUM):
        hist_v[j].append(float(u_current[2 * j]))
        hist_w[j].append(float(u_current[2 * j + 1]))

    pair_idx = 0
    for ri in range(ROBOT_NUM - 1):
        for rj in range(ri + 1, ROBOT_NUM):
            xi = float(x_current[3 * ri])
            yi = float(x_current[3 * ri + 1])
            xj = float(x_current[3 * rj])
            yj = float(x_current[3 * rj + 1])
            actual_dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            desired_dist = df_star[pair_idx]
            hist_err[pair_idx].append(actual_dist - desired_dist)
            pair_idx += 1

    # 更新动画绘图
    cargo_x, cargo_y = [], []
    x_c, y_c = 0.0, 0.0
    current_robots_pos = []
    for j in range(ROBOT_NUM):
        xr = float(x_current[3 * j])
        yr = float(x_current[3 * j + 1])
        th = float(x_current[3 * j + 2])
        x_c += xr
        y_c += yr
        verts = compute_vertices(LENGTH, WIDTH, xr, yr, th)
        robot_drawers[j].set_data(verts[0], verts[1])
        traj_data[2 * j].append(xr)
        traj_data[2 * j + 1].append(yr)
        traj_drawers[j].set_data(traj_data[2 * j], traj_data[2 * j + 1])
        cargo_x.append(xr)
        cargo_y.append(yr)
        current_robots_pos.append([xr, yr, th])

    cargo_x.append(cargo_x[0])
    cargo_y.append(cargo_y[0])
    cargo_drawer.set_data(cargo_x, cargo_y)
    traj_c[0].append(x_c / ROBOT_NUM)
    traj_c[1].append(y_c / ROBOT_NUM)
    traj_c_drawer.set_data(traj_c[0], traj_c[1])
    formation_circle.center = (x_c / ROBOT_NUM, y_c / ROBOT_NUM)
    navi_marker.set_data([navi_points[index_navi][0]],
                         [navi_points[index_navi][1]])

    # 记录stage数据（每50帧）
    if it % 50 == 0:
        stage_snapshots.append([[p[0], p[1], p[2]] for p in current_robots_pos])
        stage_data = {
            'robots': [list(p) for p in current_robots_pos],
            'cargo_poly_x': list(cargo_x),
            'cargo_poly_y': list(cargo_y),
            'centroid': (x_c / ROBOT_NUM, y_c / ROBOT_NUM),
            'traj_index': len(traj_data[0]),
        }
        all_stages_data.append(stage_data)

    plt.pause(0.001)

    if it % 50 == 0:
        elapsed = time.time() - time_start
        print(f"  Iter {it:4d} | WP {index_navi}/{len(navi_points) - 1} | "
              f"Dist={math.sqrt(dist_to_navi):.1f} | "
              f"Time={elapsed:.1f}s")
        if SAVE:
            fpath = os.path.join(save_dir, f'stage_{it // 50:03d}.png')
            fig_anim.savefig(fpath, bbox_inches='tight')

    it += 1

# 记录最终状态
final_cargo_x = [float(x_current[3*j]) for j in range(ROBOT_NUM)] + [float(x_current[0])]
final_cargo_y = [float(x_current[3*j+1]) for j in range(ROBOT_NUM)] + [float(x_current[1])]
stage_snapshots.append([[float(x_current[3*j]), float(x_current[3*j+1]), float(x_current[3*j+2])] for j in range(ROBOT_NUM)])
final_stage = {
    'robots': [[float(x_current[3*j]), float(x_current[3*j+1]), float(x_current[3*j+2])] for j in range(ROBOT_NUM)],
    'cargo_poly_x': final_cargo_x,
    'cargo_poly_y': final_cargo_y,
    'centroid': (float(cal_xc(params, x_current)[0]), float(cal_xc(params, x_current)[1])),
}
all_stages_data.append(final_stage)

# ===========================================================================
# Part 7: 仿真结束统计与图像保存
# ===========================================================================

time_end = time.time()
avg_time = (time_end - time_start) / max(it, 1)
print(f"\n{'=' * 60}")
print(f"  Simulation done  |  Iter: {it}  |  Avg: {avg_time:.4f}s/step  "
      f"|  Total: {time_end - time_start:.1f}s")
print(f"{'=' * 60}")

# 组合所有stage的图像
fig_combined, ax_combined = plt.subplots(figsize=(14, 9))
draw_all_stages_combined(ax_combined, all_stages_data, obstacles_orig,
                         ROWS, COLS, start_pt, goal_pt, traj_data, colors)
if SAVE:
    fig_combined.savefig(os.path.join(save_dir, 'all_stages_combined.png'),
                         bbox_inches='tight')
    print("  -> Saved all_stages_combined.png")
plt.close(fig_combined)

# 保存最终帧
if SAVE:
    fig_anim.savefig(os.path.join(save_dir, 'final.png'), bbox_inches='tight')
    print("  -> Saved final.png")

# 图像2：障碍物 + A*路径 + 优化轨迹 + stage快照
fig2, ax2 = plt.subplots(figsize=(14, 9))
draw_paths_and_stages(ax2, obstacles_orig, path_simplified, traj_opt, raw_path,
                      ROWS, COLS, start_pt, goal_pt, cargo_center, formation_radius,
                      stage_snapshots, traj_data, colors)
if SAVE:
    fig2.savefig(os.path.join(save_dir, 'paths_and_stages.png'), bbox_inches='tight')
    print("  -> Saved paths_and_stages.png")
plt.close(fig2)

# 图像3：安全走廊 + 规划路径总览
fig3, ax3 = plt.subplots(figsize=(14, 9))
draw_trajectory_overview(ax3, corridor, obstacles_avoid, obstacles_orig,
                         grid, ROWS, COLS, formation_radius,
                         path_simplified, traj_opt, raw_path,
                         start_pt, goal_pt, cargo_center)
if SAVE:
    fig3.savefig(os.path.join(save_dir, 'trajectory.png'), bbox_inches='tight')
    print("  -> Saved trajectory.png")
plt.close(fig3)

plt.ioff()

# ===========================================================================
# Part 8: 性能评价指标图
# ===========================================================================

draw_performance_plots(save_dir, time_hist, hist_v, hist_w, v_max_limit, w_max_limit,
                       hist_err, pair_labels, ROBOT_NUM, colors)

plt.show()
