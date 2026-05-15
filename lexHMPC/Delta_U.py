import casadi as cs

class Delta_U():

    def __init__(self, params):

        self.params = params
        self.obj_func = None

    def obj_setup(self, u):
        '''
        set up the objective function for navigation mpc problem.
        :param u: inputs
        :return: cost_delta_u
        '''
        N = self.params['N']
        robot_num = self.params['robot_num']

        cost_du = 0
        for i in range(N - 2):
            for j in range(robot_num):
                v   = u[2 * j + 0, i]
                v_n = u[2 * j + 0, i + 1]
                w   = u[2 * j + 1, i]
                w_n = u[2 * j + 1, i + 1]
                cost_du += (w - w_n) ** 2 + (v - v_n) ** 2
        # for i in range(N - 1):
        #     for j in range(robot_num):
        #         w = u[2 * j + 1, i]
        #         cost_du += w ** 2

        self.obj_func = cost_du
        return cost_du
