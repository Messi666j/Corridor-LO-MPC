import copy
import casadi as cs
from scipy.spatial import ConvexHull
import numpy as np

class HMPC():
    """
    keeps a universal list of constraints, lb, ub, respectively
    """
    def __init__(self, params, tasks):
        """
        arrange task priority as a list of levels
        :param params:
        :param tasks:
        """
        self.params = params
        self.tasks = tasks
        self.constraints = [] # universal constraints that can be used in all levels of problem
        self.lbg = []
        self.ubg = []
        self.lbx = []
        self.ubx = []

    def universal_constraints(self, x, u, lam):
        """
        add dynamics, velocity, obstacle avoidance constraints to the constraints list
        only called once at the beginning
        :param x:
        :param u:
        :param lam:
        :return: tuple of constraints, ub, lb
        """
        self.constraints = []  # universal constraints that can be used in all levels of problem
        self.lbg = []
        self.ubg = []
        self.lbx = []
        self.ubx = []

        robot_num = self.params['robot_num']
        N = self.params['N']
        v_max = self.params['v_max']
        w_max = self.params['w_max']
        dt = self.params['dt']
        obs_num, total_side_num = compute_obs_params(self.params)
        A_list, b_list = compute_obs_coeff(self.params)

        for j in range(robot_num):
            for i in range(N - 1):
                # dynamics constraints
                v = u[2 * j + 0, i]
                w = u[2 * j + 1, i]
                theta = x[3 * j + 2, i]
                self.constraints.append(
                    x[3 * j + 0, i + 1] - (x[3 * j + 0, i] + v * cs.cos(theta) * dt))  # dx
                self.lbg.append(0)
                self.ubg.append(0)
                self.constraints.append(
                    x[3 * j + 1, i + 1] - (x[3 * j + 1, i] + v * cs.sin(theta) * dt))  # dy
                self.lbg.append(0)
                self.ubg.append(0)
                self.constraints.append(
                    x[3 * j + 2, i + 1] - (x[3 * j + 2, i] + w * dt))  # d\theta
                self.lbg.append(0)
                self.ubg.append(0)

        for j in range(robot_num):
            for i in range(N - 2):
                # dynamics constraints
                v   = u[2 * j + 0, i + 0]
                v_n = u[2 * j + 0, i + 1]
                w   = u[2 * j + 1, i + 0]
                w_n = u[2 * j + 1, i + 1]
                self.constraints.append((v_n - v) / dt)  # acceleration constraints
                self.lbg.append(-v_max)
                self.ubg.append(v_max)
                # self.constraints.append(w_n - w)  # angular acceleration constraints
                # self.lbg.append(0.1)
                # self.ubg.append(0.1)

        for i in range(N):
            # obstacle constraints
            xc = cal_xc(self.params, x[:, i])
            lam_index_start = 0
            lam_index_end = 0
            for j in range(obs_num):
                A = A_list[j]
                b = b_list[j]
                side_num = A.shape[0]
                lam_index_end += side_num
                self.constraints.append(cs.transpose(A @ xc - b) @ lam[lam_index_start : lam_index_end, i])
                self.lbg.append(0.1)
                self.ubg.append(cs.inf)

                # dual norm constraint, dual norm of 2 norm is 2 norm
                self.constraints.append(cs.sumsqr(cs.transpose(A_list[j]) @ lam[lam_index_start : lam_index_end, i]))
                self.lbg += [0]
                self.ubg += [1]

                lam_index_start = lam_index_end
            # self.constraints.append(cs.dot(A.T @ lam[:, i], A.T @ lam[:, i]))
            # self.lb += [0]
            # self.ub += [1]


            # self.constraints.append(lam[:, i])
            # self.lb += [0] * total_side_num
            # self.ub += [cs.inf] * total_side_num
        # variable constraints
        self.lbx = [-cs.inf] * (3 * robot_num * N) + [-v_max, -w_max] * (robot_num * (N - 1)) + [0] * (total_side_num * N)
        self.ubx = [cs.inf] * (3 * robot_num * N) + [v_max, w_max] * (robot_num * (N - 1)) + [cs.inf] * (total_side_num * N)

        return self.constraints, self.lbg, self.ubg, self.lbx, self.ubx

    def separate_collision_avoidance(self):
        """
        collision avoidance constraints for each robot
        :return:
        """

    def moveto_cargo(self, x, u, constraints, lbg, ubg, lbx, ubx, last_solved = None):
        '''
        robots move to corresponding vertices of cargo without collision avoidance
        :param ub:
        :param lb:
        :param constraints:
        :param x: 3 * robot_num * N
        :param u: 2 * robot_num * N
        :param last_solved: last solved x and u
        :return:
        '''
        cost_navi = 0
        N = self.params['N']
        robot_num = self.params['robot_num']
        cargo = self.params['cargo']

        for i in range(N):
            for j in range(robot_num):
                cost_navi += cs.sumsqr(x[3 * j : 3 * j + 2, i] - cargo[j])

        return solve_nlp(self.params, x, u, cost_navi, constraints, lbg, ubg, lbx, ubx, last_solved)


##############################################################################################################
# end of HMPC class
##############################################################################################################

# utility functions

def start_constraint(params, x, x_current, u, u_current, constraints, lbg, ubg):
    '''
    add start constraints to the constraints list
    x0 = xs
    :param x: cs.sy
    :param x_current: numeric value
    :return:
    '''
    robot_num = params['robot_num']
    v_max = params['v_max']
    dt = params['dt']
    # deep copy of lists to avoid changing the original lists
    constraints = copy.deepcopy(constraints)
    lbg = copy.deepcopy(lbg)
    ubg = copy.deepcopy(ubg)

    for j in range(3 * robot_num): # initial states
        constraints.append(x[j, 0] - x_current[j])
        lbg.append(0)
        ubg.append(0)

    for j in range(robot_num): # initial inputs
        constraints.append((u[2 * j, 0] - u_current[2 * j]) / dt)
        lbg.append(-v_max)
        ubg.append(v_max)

    return constraints, lbg, ubg

def construct_x0(params, states):
    '''
    from last solved states, construct initial guess x0 for next iteration
    :param params: necessary parameters
    :param states: last solved states, DM variable,
    shape of [3 * robot_num * N + 2 * robot_num * (N - 1)]
    :return: initial guess x0, shape of [3 * robot_num * N + 2 * robot_num * (N - 1)]
    '''
    robot_num = params['robot_num']
    N = params['N']
    obs_num, total_side_num = compute_obs_params(params)

    states_index_start = 3 * robot_num
    states_index_end = 3 * robot_num * N
    input_index_start = 3 * robot_num * N + 2 * robot_num
    input_index_end = 3 * robot_num * N + 2 * robot_num * (N - 1)
    lam_index_start = 3 * robot_num * N + 2 * robot_num * (N - 1) + total_side_num
    lam_index_end = 3 * robot_num * N + 2 * robot_num * (N - 1) + total_side_num * N
    # x0 = x1 : xN
    x0 = states[states_index_start : states_index_end]
    # repeat xN
    x0 = cs.vertcat(x0, states[states_index_end - states_index_start : states_index_end])
    # u1: uN-1
    x0 = cs.vertcat(x0, states[input_index_start : input_index_end])
    # concat 0
    x0 = cs.vertcat(x0, cs.DM.zeros(2 * robot_num))
    # lam1 : lamN
    x0 = cs.vertcat(x0, states[lam_index_start : lam_index_end])
    # repeat lamN
    x0 = cs.vertcat(x0, states[lam_index_end - total_side_num : lam_index_end])

    # x0 shape should be [3 * robot_num * N + 2 * robot_num * (N - 1) + total_side_num * N]
    return x0

def update_state(params, x_current, u_current):
    # xc = self.cal_xc(self.x_current)
    # distance = float(cs.norm_2(xc - self.c_obs))
    # self.TOLERANCE_AVOID = 5e-3 * distance * self.N
    robot_num = params['robot_num']
    dt = params['dt']
    for j in range(robot_num):
        # input reference
        v = u_current[2 * j + 0]
        w = u_current[2 * j + 1]
        # states reference
        theta = x_current[3 * j + 2]

        x_current[3 * j + 0] = x_current[3 * j + 0] + v * cs.cos(theta) * dt
        x_current[3 * j + 1] = x_current[3 * j + 1] + v * cs.sin(theta) * dt
        x_current[3 * j + 2] = x_current[3 * j + 2] + w * dt

def extract_input(solved, x_index_end, u_current_index_end):
    '''
    extract input from solved (x, u) vector
    :param solved:
    :param x_index_end:
    :param u_current_index_end:
    :return:
    '''
    return solved[x_index_end : x_index_end + u_current_index_end]

def cal_xc(params, states):
    '''
    :param states: x_current
    :return:
    '''
    robot_num = params['robot_num']
    xc = cs.SX([0, 0])
    for j in range(robot_num):
        xc[0] += states[3 * j + 0]
        xc[1] += states[3 * j + 1]
    xc /= robot_num

    return xc
def add_lex_constraint(cost_func, f_val, epsilon, constraints, lbg, ubg):
    """
    add lex constraint to the constraints list
    :param cost_func:
    :param f_val:
    :param constraints:
    :param lbg:
    :param ubg:
    :param epsilon:
    :return:
    """

    constraints.append(cost_func - f_val)
    lbg.append(-cs.inf)
    ubg.append(epsilon)

    return constraints, lbg, ubg

def compute_obs_params(params):
    """
    compute the obs_num and total_side_num for obstacles
    :param params:
    :return: (obs_num, total_side_num)
    """
    obstacles = params['obstacles_avoid']
    obs_num = len(obstacles)
    total_side_num = 0
    for i in range(obs_num):
        total_side_num += len(obstacles[i])

    return obs_num, total_side_num

def compute_obs_coeff(params):
    """
    given vertices of obstacles, compute the corresponding A,b
    2d, so only consider x, y
    :param params:
    :return: list of A, b
    """

    obstacles = params['obstacles_avoid']

    A_list = []
    b_list = []

    for obs_vertices in obstacles:
        centroid = np.mean(obs_vertices, axis=0)
        hull = ConvexHull(obs_vertices)
        # the centroid should be inside the hull, satisfying ax <= b
        A = cs.DM(hull.equations[:, :2])
        b = cs.DM(-hull.equations[:, 2])
        A_list.append(A)
        b_list.append(b)

    return A_list, b_list



def solve_nlp(params, x, u, lam, cost, constraints, lbg, ubg, lbx, ubx, last_solved = None):
    '''
    given cost, build a nlp solver and solve it
    :param last_solved:
    :param u:
    :param x:
    :param params:
    :param cost:
    :param constraints:
    :param lb:
    :param ub:
    :return:
    '''
    # extract parameters
    robot_num = params['robot_num']
    N = params['N']
    obs_num, total_obs_side = compute_obs_params(params)

    X = cs.reshape(x, -1, 1)
    U = cs.reshape(u, -1, 1)
    Lam = cs.reshape(lam, -1, 1)
    # construct initial guess
    if last_solved is not None:
        x0 = construct_x0(params, last_solved)
        # x0 = last_solved
        # x0[3 * robot_num * N: ] = 0
    else:
        x0 = [0] * (3 * robot_num * N + 2 * robot_num * (N - 1) + total_obs_side * N)
    # set up the nlp problem and solve it
    nlp = {'x': cs.vertcat(X, U, Lam), 'f': cost, 'g': cs.vertcat(*constraints)}
    opts = {'ipopt.print_level': 0, 'print_time': 0}
            # 'ipopt.acceptable_constr_viol_tol': 1e-10,
            # 'ipopt.bound_relax_factor': 0,
            # 'ipopt.constr_viol_tol': 1e-10,
            # 'ipopt.acceptable_compl_inf_tol': 1e-10,
            # 'ipopt.compl_inf_tol': 1e-10}
    # S = cs.nlpsol('S', 'ipopt', nlp)
    S = cs.nlpsol('S', 'ipopt', nlp, opts)

    solution = S(x0=x0, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
    # solution = S(lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)

    return solution