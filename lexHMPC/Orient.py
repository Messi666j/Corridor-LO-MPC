import casadi as cs
from lexHMPC.HMPC import cal_xc, construct_x0, solve_nlp


class Orientation():
    '''
    solves a single orientation MPC problem
    '''

    def __init__(self, params):

        self.params = params
        self.obj_func = None

    def obj_setup(self, x, a, theta_d, l):
        '''
        set up the objective function for navigation mpc problem.
        :param theta_d: desired orientation
        :param a: weight vector of the orientation
        :param x: state
        :return: cost_navi
        '''
        N = self.params['N']
        # only considers the last time index
        xc = cal_xc(self.params, x[:, N - 1])
        xh = self.cal_xh(a, x[:, N - 1])
         # cs.norm_2(xc - xh) # l is |xh - xc|
        cost_orient = (xh[1] - xc[1] - l * cs.sin(theta_d)) ** 2 + \
                       (xh[0] - xc[0] - l * cs.cos(theta_d)) ** 2

        self.obj_func = cost_orient

        return cost_orient

    def cal_xh(self, a, x_current):
        """
        calculate the reference point for the orientation MPC problem
        :param x_current:
        :return:
        """
        xh = cs.SX.zeros(2)
        robot_num = self.params['robot_num']
        for i in range(robot_num):
            xh[0] += a[i] * x_current[3 * i] # x
            xh[1] += a[i] * x_current[3 * i + 1] # y

        return xh

    def cal_orient_err(self, x_current, a, theta_d):
        """
        calculate the orientation error
        :param x:
        :param a:
        :param theta_d:
        :return:
        """
        xh = self.cal_xh(a, x_current)
        xc = cal_xc(self.params, x_current)
        # return cs.atan2(xh[1] - xc[1], xh[0] - xc[0]) - theta_d
        import math
        return math.atan2(xh[1] - xc[1], xh[0] - xc[0]) - theta_d
