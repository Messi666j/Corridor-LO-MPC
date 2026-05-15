import numpy as np
import matplotlib.pyplot as plt


def compute_vertices(length, width, x, y, theta):
    '''
    compute vertices of a robot or cargo with given length, width and states
    :param length:
    :param width:
    :param x_current: the coordinates of x_current.
    :param x: the x coordinate of the robot.
    :param y: the y coordinate of the robot.
    :param theta: the angle of the robot.
    :return:
    '''
    phi1 = np.pi / 2 - np.arctan(length / width)
    phi2 = np.pi - phi1
    B = np.linalg.norm([length, width]) / 2
    vertices_x = [x + B * np.cos(theta + phi1),
                  x + B * np.cos(theta + phi2),
                  x + B * np.cos(theta - phi2),
                  x + B * np.cos(theta - phi1),
                  x + B * np.cos(theta + phi1)]
    vertices_y = [y + B * np.sin(theta + phi1),
                  y + B * np.sin(theta + phi2),
                  y + B * np.sin(theta - phi2),
                  y + B * np.sin(theta - phi1),
                  y + B * np.sin(theta + phi1)]
    vertices = [vertices_x, vertices_y]
    return vertices


def draw_robot(ax, params, x_current, colors):
    '''
    set up robot drawers， traj_drawers and traj[] for storing coordinates
    :param ax:
    :param states: 3 * 1
    :param length:
    :param width:
    :return:
    '''
    robot_num = params['robot_num']
    length = params['length']
    width = params['width']
    robot_drawers = []
    traj = []
    traj_drawers = []
    traj_c = [[0], [0]]
    for j in range(robot_num):
        x = float(x_current[3 * j + 0])
        y = float(x_current[3 * j + 1])
        theta = float(x_current[3 * j + 2])
        traj_c[0][0] += x
        traj_c[1][0] += y
        vertices = compute_vertices(length, width, x, y, theta)
        robot_drawer, = ax.plot(vertices[0], vertices[1], color=colors[j], label="Vehicle {} ".format(j+1))
        robot_drawers.append(robot_drawer)

        traj.append([x])
        traj.append([y])
        traj_drawer, = ax.plot(x, y, color=colors[j], linestyle='--')
        traj_drawers.append(traj_drawer)

    traj_c[0][0] = traj_c[0][0] / robot_num
    traj_c[1][0] = traj_c[1][0] / robot_num
    traj_c_drawer, = ax.plot(traj_c[0], traj_c[1], label="Centroid", color='darkred')
    return robot_drawers, traj_drawers, traj, traj_c_drawer, traj_c


def draw_cargo(ax, params):
    '''
    set up cargo drawer
    :param ax:
    :param params:
    :param x_current:
    :param colors:
    :return:
    '''
    cargo = params['cargo']
    cargo_x = []
    cargo_y = []
    for j in range(-1, len(cargo)):
        cargo_x.append(cargo[j][0])
        cargo_y.append(cargo[j][1])
    cargo_drawer = ax.plot(cargo_x, cargo_y, color='k')

    return cargo_drawer


def draw_obstacle(ax, params):
    '''
    draw obstacles
    :param ax:
    :param params:
    :return:
    '''

    if params['obs_type'] == 'polygonal':
        obstacles = params['obstacles']
        for obstacle in obstacles:
            left_under = obstacle[0]
            width = obstacle[-1][0] - obstacle[0][0]
            height = obstacle[1][1] - obstacle[0][1]
            rect = plt.Rectangle(left_under, width, height, angle=0, color='k')
            ax.add_patch(rect)

    else:
        obs_o = params['obs_o']
        obs_r = params['obs_r']
        for i in range(len(obs_o)):
            ax.add_patch(plt.Circle(obs_o[i], obs_r[i], color='k'))


def draw_destination(ax, params):
    destination = params['destination']
    ax.plot(float(destination[0]), float(destination[1]), marker='x', color='k')


def draw_navi_points(ax, navi_points):
    x = []
    y = []
    for j in range(len(navi_points)):
        x.append(navi_points[j][0])
        y.append(navi_points[j][1])

    navi_points_drawer = ax.scatter(x, y, color='r')

    return navi_points_drawer
