import casadi as cs
from lexHMPC.HMPC import cal_xc, construct_x0, solve_nlp
import numpy as np


class Navigation():
    '''
    solves a single obstacle avoidance MPC problem
    '''

    def __init__(self, params):
        self.params = params
        self.obj_func = None

    def obj_setup(self, x, x_current, destination, a, theta_d, l):
        '''
        set up the objective function for navigation mpc problem.
        now including orientation.
        :param x: state
        :return: cost_navi
        '''
        N = self.params['N']
        # destination = self.params['destination']

        # xc = cal_xc(self.params, x[:, N - 1])
        # # only considers the xN
        # cost_navi = cs.sumsqr(xc - destination)

        cost_navi = 0
        # navigation
        for i in range(N):
            xc = cal_xc(self.params, x[:, i])
            cost_navi += cs.sumsqr(xc - destination)
        # orientation
        if (destination == self.params['destination']).all():
            coeff = cs.norm_2(cal_xc(self.params, x_current) - destination) / l  # weights of orientation and navigation are 1:1
            for i in range(N):
                xh = cal_xh(self.params, a, x[:, i])
                xc = cal_xc(self.params, x[:, i])
                cost_navi += coeff * ((xh[1] - xc[1] - l * cs.sin(theta_d)) ** 2 + (xh[0] - xc[0] - l * cs.cos(theta_d)) ** 2)
        self.obj_func = cost_navi

        return cost_navi


def cal_xh(params, a, x_current):
    """
    calculate the reference point for the orientation MPC problem
    :param x_current:
    :return:
    """
    xh = cs.SX.zeros(2)
    robot_num = params['robot_num']

    for i in range(robot_num):
        xh[0] += a[i] * x_current[3 * i]  # x
        xh[1] += a[i] * x_current[3 * i + 1]  # y

    return xh


def cal_l(params, a, cargo):
    """
    :param params:
    :param a:
    :param cargo:
    :return: l the norm of the vector from the reference point to the center
    """
    xc = np.mean(cargo, axis=0)
    a = np.array(a).reshape(1, -1)
    cargo = np.array(cargo)
    xh = a @ cargo
    l = cs.norm_2(xc - xh)

    return l
