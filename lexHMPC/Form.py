import casadi as cs
from lexHMPC.HMPC import solve_nlp


class Formation():
    '''
    solves a single formation MPC problem
    '''

    def __init__(self, params):

        self.params = params
        self.obj_func = None

    def cal_err_form(self, x):
        '''
        calculate the error of the formation
        :param states:
        :return:
        '''
        # extract parameters
        robot_num = self.params['robot_num']
        df_star = self.params['df_star']

        df_index = 0
        err_form = 0
        for j in range(robot_num - 1):
            for k in range(j + 1, robot_num):
                err_form += (cs.sumsqr(x[3 * j: 3 * j + 2] - x[3 * k: 3 * k + 2])
                             - df_star[df_index] ** 2) ** 2
                df_index += 1
        return err_form

    def obj_setup(self, x):
        """
        set up the objective function for formation mpc problem.
        :param x:
        :return:
        """
        N = self.params['N']

        cost_form = 0
        for i in range(N):
            cost_form += self.cal_err_form(x[:, i])

        self.obj_func = cost_form
        return cost_form




