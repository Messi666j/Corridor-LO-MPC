# Corridor-LO-MPC

## 项目简介

`Corridor-LO-MPC` 是一个面向多机器人编队导航与避障的仿真项目，结合了 `LO-MPC`、`Convex Safe Corridor`、`A*` 路径规划与 `minimum snap` 轨迹优化，用于在障碍环境中生成可行路径、构建安全走廊，并实现编队控制与避障仿真。

项目的整体流程从栅格地图构建开始，经过初始路径搜索、路径简化、走廊生成与轨迹平滑，最终由 LO-MPC 完成多机器人编队运动控制。运行后可输出轨迹图、走廊图、阶段图、速度曲线与编队误差等结果，便于观察和分析仿真过程。

## 核心特性

- 支持在栅格地图中定义障碍物环境并进行仿真
- 基于 `A*` 搜索生成初始可行路径
- 对原始路径进行简化，提取关键导航点
- 基于路径生成 `Convex Safe Corridor`
- 使用 `minimum snap` 方法对路径进行轨迹优化
- 基于 `LO-MPC` 实现多机器人编队控制与避障
- 自动输出走廊图、轨迹图、阶段图、速度图和误差图等结果

## 方法流程 / Pipeline

1. 构建二维栅格地图，并定义障碍物区域
2. 对障碍物进行安全膨胀，生成适合编队通行的安全地图
3. 使用 `A*` 搜索从起点到目标点的初始路径
4. 对路径进行简化，提取关键路径点
5. 根据简化路径生成 `Convex Safe Corridor`
6. 在安全走廊内进行 `minimum snap` 轨迹优化
7. 将优化结果作为导航参考，使用 `LO-MPC` 进行编队控制
8. 输出仿真图像与相关结果用于分析

## 项目结构

```text
Corridor-LO-MPC/
├── README.md
├── pyproject.toml
├── uv.lock
├── main.py
├── lexHMPC/
│   ├── HMPC.py
│   ├── Form.py
│   ├── Navi.py
│   ├── Orient.py
│   ├── Delta_U.py
│   ├── Avoid.py
│   ├── Moveto_Cargo.py
│   └── ...
├── utils/
│   ├── ConvexCorridor2.py
│   ├── MapBuilder.py
│   ├── CorridorDrawing.py
│   ├── Drawing.py
│   ├── Astar.py
│   ├── VisibilityGraph.py
│   └── ...
├── simulation/
│   ├── Simulation1.py
│   └── output/
└── output/
    └── ...
```

## 环境要求

- Python `>= 3.13`
- 使用 `pyproject.toml` 管理依赖
- 主要依赖包括：
  - `casadi`
  - `cvxpy`
  - `matplotlib`
  - `numpy`
  - `opencv-python`
  - `scipy`
  - `shapely`

## 安装方式

推荐使用 `uv` 安装和管理项目依赖。

```bash
cd /path/to/Corridor-LO-MPC
uv sync
```

如果你当前就在项目目录下，可直接执行：

```bash
uv sync
```

## 运行方法

运行主仿真脚本：

```bash
uv run python main.py
```

运行示例仿真脚本：

```bash
uv run python simulation/Simulation1.py
```

## 结果展示

### 主仿真效果

![Final Result](output/main/final.png)

### 轨迹结果

![Trajectory](output/main/trajectory.png)

### 安全走廊结果

![Corridor](output/main/corridor_only.png)

## 关键参数说明

以下参数可在脚本中根据实验需求进行调整：

- `obstacle_regions`  
  用于定义障碍物区域布局，决定栅格地图中的障碍配置。

- `ROBOT_NUM`  
  编队中的机器人数量，用于控制系统规模与编队状态维度。

- `destination`  
  编队导航的目标位置。

- `N`  
  MPC 预测时域长度，影响控制优化的前瞻性与计算规模。

- `dt`  
  离散时间步长，影响仿真的时间分辨率与控制更新频率。

## 输出说明

运行完成后，结果通常会保存在 `output/` 和 `simulation/output/` 目录下，内容包括但不限于：

- 安全走廊可视化图
- 原始路径、简化路径与优化轨迹图
- 分阶段仿真结果图
- 编队整体轨迹总览图
- 线速度与角速度相关曲线
- 编队误差图
- 其他中间结果或实验输出文件

这些结果可用于展示路径规划效果、编队控制过程以及整体避障表现。

## 后续改进方向

- 增加更多场景配置，支持更复杂的障碍布局
- 补充统一的参数配置入口，减少直接修改脚本的成本
- 增强动态障碍物场景的支持与实验组织方式
- 完善模块说明与算法注释，提升代码可读性
- 增加更多可复现实验脚本，便于结果对比与展示