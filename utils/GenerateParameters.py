import numpy as np
import math
import cv2 as cv
import copy
from lexHMPC import Navi, HMPC

def getConnectDomain(binary):
    '''
    :param binary: binary image
    :return: number of connectivity domain
    '''
    contours, _ = cv.findContours(binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    return len(contours)


def getVertex(binary):
    '''
    :param binary: binary image
    :return:
    A list of contours found in the image.
    Each contour is represented as an array of points, which outlines the shape of the detected object.
    '''
    contours, _ = cv.findContours(binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    polygons = []
    for i in range(len(contours)):
        polygon = []
        for j in range(len(contours[i])):
            if abs(sum(contours[i][j][0] - contours[i][j - 1][0])) < 6:
                continue
            polygon.append(contours[i][j][0])
        polygons.append(polygon)
    return polygons


def getCenter(binary):  # 连通域中心
    '''

    :param binary: image
    :return: number of connectivity domain, a list of the centers of the connectivity domains
    '''
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(binary, connectivity=8)
    return num_labels-1, np.delete(centroids, 0, axis=0)  # 中心点坐标


def getRadius(binary):  # 最小外接圆半径
    '''
    Determine the avoidance radius of the formation and the radius of the circular obstacle.
    :param binary: image
    :return: the radius of the minimum enclosing circle
    '''
    contours, _ = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    coordinate = []
    radius = []
    for cnt in contours:
        [x, y], r = cv.minEnclosingCircle(cnt)
        coordinate.append([x, y])
        radius.append(r)
    return coordinate, radius


def getArea(obs):  # 障碍物面积
    '''
    Calculate the area of the polygonal obstacle.
    :param obs: vertices of the polygonal obstacle (in clockwise order)
    :return: area
    '''
    area = 0
    for i in range(len(obs)):
        area += obs[i-1][0] * obs[i][1] - obs[i-1][1] * obs[i][0]
    return abs(area) / 2


def sim_params(img, obs_type = 'polygonal'):
    obstacles_avoid = None
    obs_o = None
    obs_r = None
    img = np.flipud(img)
    # The color representation in OpenCV is BGR
    obstacles_l = np.array([0, 0, 0])  # the BGR of the obstacles is [0, 0, 0], which is black
    obstacles_u = np.array([0, 0, 0])
    obstacles_dyna_u = np.array([127, 127, 127])  # the BGR of the dynamic obstacles is [127, 127, 127], which is gray
    obstacles_dyna_l = np.array([127, 127, 127])
    robot_l = np.array([255, 0, 0])  # the BGR of the robots is [255, 0, 0], which is blue
    robot_u = np.array([255, 0, 0])
    final_l = np.array([0, 0, 255])  # the BGR of the final point is [0, 0, 255], which is red
    final_u = np.array([0, 0, 255])
    cargo_l = np.array([0, 255, 0])  # the BGR of the cargo is [0, 255, 0], which is green
    cargo_u = np.array([0, 255, 0])

    img_obs = cv.inRange(img, obstacles_l, obstacles_u)
    img_obs_dyna = cv.inRange(img, obstacles_dyna_l, obstacles_dyna_u)
    img_robot = cv.inRange(img, robot_l, robot_u)
    img_final = cv.inRange(img, final_l, final_u)
    img_cargo = cv.inRange(img, cargo_l, cargo_u)
    # cv.imshow('obs', img_obs)
    # cv.waitKey(0)

    robot_num, robots = getCenter(img_robot)  # 机器人中心点坐标
    robots = robots.flatten()

    polygons = getVertex(img_robot)
    theta = []  # 机器人朝向
    for polygon in polygons:
        if np.linalg.norm(polygon[0] - polygon[1], 2) > np.linalg.norm(polygon[1] - polygon[2], 2):
            theta.append(math.pi / 2 if -0.001 < polygon[0][0] - polygon[1][0] < 0.001
                         else math.atan((polygon[0][1] - polygon[1][1]) / (polygon[0][0] - polygon[1][0])))
        else:
            theta.append(math.pi / 2 if -0.001 < polygon[2][0] - polygon[1][0] < 0.001
                         else math.atan((polygon[2][1] - polygon[1][1]) / (polygon[2][0] - polygon[1][0])))
    theta = np.array(theta)  # 机器人朝向
    for i in range(len(theta)):   # [x_1, y_1, theta_1, ..., x_n, y_n, theta_n]
        robots = np.insert(robots, i * 3 + 2, theta[i])

    l1 = np.linalg.norm(polygons[0][0] - polygons[0][1])
    l2 = np.linalg.norm(polygons[0][2] - polygons[0][1])
    if l1 > l2:  # 机器人参数，见diff_robot_params
        length, width = l1, l2
    else:
        length, width = l2, l1

    cargo = getVertex(img_cargo)  # 货物顶点坐标
    cargo = np.squeeze(cargo)
    cargo_center, cargo_radius = getRadius(img_cargo)
    cargo_radius = cargo_radius[0]

    obs_num = getConnectDomain(img_obs)  # 障碍物数量
    if obs_type == 'polygonal':
        obstacles = getVertex(img_obs)
        obstacles_avoid = copy.deepcopy(obstacles)
        avoid_r = int(cargo_radius)
        for i in range(len(obstacles_avoid)):
            obstacles_avoid[i][0][0] -= avoid_r
            obstacles_avoid[i][0][1] -= avoid_r
            obstacles_avoid[i][1][0] -= avoid_r
            obstacles_avoid[i][1][1] += avoid_r
            obstacles_avoid[i][2][0] += avoid_r
            obstacles_avoid[i][2][1] += avoid_r
            obstacles_avoid[i][3][0] += avoid_r
            obstacles_avoid[i][3][1] -= avoid_r

        kernel = np.ones((int(cargo_radius + 2 * length), int(cargo_radius + 2 * length)), np.uint8)
        img_obs_dilated = cv.dilate(img_obs, kernel, iterations=1)  # dilate obstacles for generating navigation points
        obstacles_navi_point = getVertex(img_obs_dilated)  # vertices of polygonal obstacles

        obstacles_dyna = getVertex(img_obs_dyna)
        obstacles_dyna_center, obstacles_dyna_radius = getRadius(img_obs_dyna)
    else:
        obstacle_center, r = getRadius(img_obs)  # circular obstacles
        obstacle_radius = [r[i] + cargo_radius + width for i in range(len(r))]

    _, pd = getCenter(img_final)  # 终点坐标
    pd = pd.flatten()

    df_star = get_df_star(cargo)

    imgs = {
        'img_obs': img_obs,
        'img_obs_dyna': img_obs_dyna
    }

    simulation_params = {
        'robot_num': robot_num,  # number of robot
        'robots': robots,  # coordinates of the robots
        'length': length,
        'width': width,
        'destination': pd,  # destination
        'obs_type': obs_type,
        'obs_num': obs_num,
        'cargo': cargo,
        'cargo_radius': cargo_radius,
        'cargo_center': cargo_center,
        'df_star': df_star,
        'N': 4,  # prediction horizon
        'dt': 0.1,  # time interval
        'v_max': width,
        'w_max': 1.0,
        'theta_d': -math.pi/2,
        'a': [0, 0.5, 0.5, 0]
    }
    if obs_type == 'polygonal':
        simulation_params['obstacles'] = obstacles  # original obstacles
        simulation_params['obstacles_avoid'] = obstacles_avoid  # dilated obstacles for avoiding
        # simulation_params['obstacles_area'] = [getArea(obstacle) for obstacle in obstacles]
        simulation_params['obstacles_navi_point'] = obstacles_navi_point  # dilated obstacles for navigation points
        simulation_params['obstacles_dyna'] = obstacles_dyna  # coordinates of the obstacles for dynamic obstacles
        simulation_params['obstacles_dyna_center'] = obstacles_dyna_center  # coordinates of center of the dynamic obstacles
        simulation_params['obstacles_dyna_radius'] = obstacles_dyna_radius  # radii of the minimum circumscribed circles of the dynamic obstacles
    else:
        simulation_params['obstacle_center'] = obstacle_center # coordinates of the obstacles
        simulation_params['obstacle_radius'] = obstacle_radius # radius of the obstacles

    return simulation_params, imgs


def get_inner_distance(points):
    '''
    points: of shape 2n * 1. e.g. [x1, y1, x2, y2, ..., xn, yn]
    '''
    x = []
    y = []
    for i in range(int(len(points) / 2)):
        x.append(points[2 * i + 0])
        y.append(points[2 * i + 1])
    inner_dist = []
    for i in range(len(x) - 1):
        for j in range(i + 1, len(x)):
            inner_dist.append(math.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2))

    return inner_dist


def get_df_star(cargo):
    cargo = np.array(cargo).flatten().tolist()
    df_star = get_inner_distance(cargo)

    return df_star

# def getposBgr(event, x, y, flags, param):  # 用于确定像素点的BGR值
#     if event == cv.EVENT_LBUTTONDOWN:
#         print("BGR is", img[y, x])
# cv.imshow('image', img)
# cv.setMouseCallback("image", getposBgr)
# cv.waitKey(0)

# generate parameters and save them in the output folder
if __name__ == '__main__':
    path = '../maps/dyna_obs.png'
    img = cv.imread(path)
    params, imgs = sim_params(img)
    print(params['obstacles_dyna_center'])

    # print(len(params['cargo']))