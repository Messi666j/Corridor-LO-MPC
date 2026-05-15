import casadi as cs
from lexHMPC.HMPC import cal_xc, construct_x0, solve_nlp


class Avoidance():
    '''
    solves a single obstacle avoidance MPC problem
    '''

    def __init__(self, params):

        self.params = params
        self.obj_func = None

    def cal_err_avoid(self, states, index_obs, barrier_function='log'):
        '''
        Calculating the objective function of the avoidance MPC problem
        :param states:
        :param index_obs: index of the obstacle
        :param barrier_function: function to use for barrier function
        :return:
        '''

        xc = cal_xc(self.params, states)
        tmp = 0  # area of the triangle if the obstacle is polygonal; distance to the center of obstacle if round.
        if self.params['obs_type'] == 'polygonal':
            obstacle = self.params['obstacles'][index_obs]
            area_obstacle = self.params['obstacles_area'][index_obs]
            for j in range(len(obstacle)):
                tmp += self.cal_area_tri(xc, obstacle[j - 1], obstacle[j])
            tmp -= area_obstacle
        else:  # round
            center_obstacle = self.params['obstacles_center'][index_obs]
            radius_obstacle = self.params['obstacles_radius'][index_obs]
            tmp = cs.sumsqr(xc - center_obstacle) - radius_obstacle ** 2

        if barrier_function == 'log':
            err_avoid = - cs.log(tmp) / 5  # t = 5
        elif barrier_function == 'inverse':
            err_avoid = 1 / tmp
        else:
            err_avoid = 1 / tmp

        return err_avoid

    def cal_area_tri(self, x1, x2, x3):
        return cs.fabs(x1[0] * (x2[1] - x3[1]) + x2[0] * (x3[1] - x1[1]) + x3[0] * (x1[1] - x2[1])) / 2

    def obj_setup(self, x):
        '''
        setting up the objective function for avoid mpc problem.
        :param x:
        :param u:
        :return:
        '''
        cost_avoid = 0
        N = self.params['N']
        obs_num = self.params['obs_num']

        for i in range(N):
            for k in range(obs_num):
                cost_avoid += self.cal_err_avoid(x[:, i], k, barrier_function='log')

        self.obj_func = cost_avoid
        return cost_avoid

