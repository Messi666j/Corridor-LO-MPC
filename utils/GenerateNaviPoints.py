from utils.Astar import astar
import copy
from lexHMPC.HMPC import cal_xc
from utils.GenerateParameters import getVertex
import numpy as np
import cv2 as cv


def get_navi_points(params):
    '''
    :param p_i:  initial point
    :param p_f: final point
    :param obss: obstacles
    :return: indices of navigation points, coordinates of the navigation points
    '''
    p_i = params['cargo_center'][0]
    p_f = params['destination']
    obss = copy.deepcopy(params['obstacles_navi_point'])
    a = astar(p_i, p_f, obss)
    index, position = a.searchPath()
    return index, position


def regenerate_navi_points(params, imgs, new_obs, x_current):
    cargo_radius = params['cargo_radius']
    length = params['length']
    img_obs = imgs['img_obs']
    img_obs_dyna = imgs['img_obs_dyna']
    img_obs[new_obs[0][1]: new_obs[2][1], new_obs[0][0]: new_obs[2][0]] = 255
    img_obs_dyna[new_obs[0][1]: new_obs[2][1], new_obs[0][0]: new_obs[2][0]] = 0
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

    params['obstacles'] = obstacles
    params['obstacles_avoid'] = obstacles_avoid

    kernel = np.ones((int(cargo_radius + 2 * length), int(cargo_radius + 2 * length)), np.uint8)
    img_obs_dilated = cv.dilate(img_obs, kernel, iterations=1)  # dilate obstacles for generating navigation points
    params['obstacles_navi_point'] = getVertex(img_obs_dilated)  # vertices of polygonal obstacles

    xc = cal_xc(params, x_current)
    p_i = [float(xc[0]), float(xc[1])]
    p_f = params['destination']
    obss = copy.deepcopy(params['obstacles_navi_point'])
    a = astar(p_i, p_f, obss)
    index, position = a.searchPath()
    return index, position
