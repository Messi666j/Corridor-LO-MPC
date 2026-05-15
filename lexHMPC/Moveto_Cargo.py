import casadi as cs
from lexHMPC import Form, Navi
from lexHMPC.HMPC import solve_nlp


class Moveto_Cargo():
    '''
    solves a single formation MPC problem
    '''

    def __init__(self, params):

        self.params = params
        self.obj_func = None

    def obj_setup(self, x):
        """
        set up the objective function for formation mpc problem.
        :param x:
        :return:
        """
        cost = 0
        cargo = self.params['cargo']
        N = self.params['N']
        robot_num = self.params['robot_num']
        for i in range(N):
            for j in range(robot_num):
                cost += cs.sumsqr(x[3 * j: 3 * j + 2, i] - cargo[j])

        return cost

    def add_avoid_constraint(self, x, constraints, lbg, ubg):
        robot_num = self.params['robot_num']
        length = self.params['length']
        width = self.params['width']
        N = self.params['N']
        # r = (length ** 2 + width ** 2) / 0.8
        r = (1.5 * length) ** 2
        N = 1

        for i in range(N):
            for j in range(robot_num - 1):
                for k in range(j + 1, robot_num):
                    constraints.append(cs.sumsqr(x[3 * j: 3 * j + 2, i] - x[3 * k: 3 * k + 2, i]))
                    lbg.append(r)
                    ubg.append(cs.inf)
        return constraints, lbg, ubg



