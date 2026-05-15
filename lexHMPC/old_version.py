import casadi as cs
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2 as cv

from lexHMPC import Avoid
from utils import GenerateParameters


class HMPC():

    def __init__(self, map):
        params = GenerateParameters.sim_params(map)
        # robot params
        self.ROBOT_NUM = params['robot_num']
        self.WMAX = 1  # max angle velocity
        self.VMAX = 20  # max velocity
        self.BETA_MAX = 0.2  # max angle acceleration
        self.LENGTH = params['length']  # length of the robot, only used to draw the robot
        self.WIDTH = params['width']  # width of the robot, only used to draw the robot
        # self.wmax = 10
        # self.l = 0.1
        # self.r = 0.5
        # self.b = 0.5

        # obstacles params
        self.obstacles = params[
            'obstacles_o']  # of shape j * 2. obstacles[j] denotes the coordinate of the jth obstacle
        self.obstacles_r = params['obstacles_r']  # radius of each obstacle
        # cargo params
        self.cargo = params[
            'cargo']  # of shape j * 2. cargo[j] denotes the coordinate of the jth vertex of the cargo. clockwise
        self.cargo_r = params['cargo_r']  # Minimum circumscribed circle of the cargo. For avoidance

        # sim params
        self.N = 4
        self.dt = 0.1
        self.x_current = cs.SX(params['robots'])  # initial states
        self.u_current = cs.SX([0, 0, 0, 0, 0, 0, 0, 0])  # initial states
        self.df_star = cs.SX(params['df_star'])  # formation params
        self.destination = cs.SX(params['destination'])
        self.err_form = None
        self.err_avoid = None
        self.err_navi = None
        self.solution = None
        self.phase = "moving_to_cargo"  # options: moving_to_cargo, forming and transporting

        self.xN = cs.SX([0] * self.ROBOT_NUM * 3)  # last predicted position
        self.it_stuck = 0

        # algorithm params
        self.x = cs.SX.sym('x', 12, self.N)
        self.u = cs.SX.sym('u', 8, self.N - 1)
        self.constraints = []
        self.lb = []
        self.ub = []
        self.virtual_obstacles = []
        self.EPSILON_FORM = (self.cargo_r ** 4) * 0.1 * self.N
        self.EPSILON_NAVI = 0.1 * self.N
        self.BARRIER_FUNCTION = 'log'  # options: inverse and log, default: inverse
        if self.BARRIER_FUNCTION == 'inverse':
            self.EPSILON_AVOID = 0.01 * self.N
        elif self.BARRIER_FUNCTION == 'log':
            self.EPSILON_AVOID = 0.1 * self.N
        else:
            self.EPSILON_AVOID = 0.01 * self.N

        # draw params
        self.fig = plt.figure()
        self.ax = self.fig.gca()
        plt.axis('equal')
        # plt.xlim((-5, 20))
        # plt.ylim((-5, 20))
        self.draw_traj = []
        self.draw_robot = []
        self.draw_cargo = []
        self.traj = []
        color = ['tomato', 'orange', 'royalblue', 'darkgreen']

        # draw robots and trajectories
        for j in range(self.ROBOT_NUM):
            x = float(self.x_current[3 * j + 0])
            y = float(self.x_current[3 * j + 1])
            theta = float(self.x_current[3 * j + 2])
            vertices = self.compute_vertices(x, y, theta)
            draw_robot, = self.ax.plot(vertices[0], vertices[1], color=color[j], label="robot {} ".format(j))
            self.draw_robot.append(draw_robot)

            self.traj.append([x])
            self.traj.append([y])
            draw_traj, = self.ax.plot(x, y, color=color[j])
            self.draw_traj.append(draw_traj)
        # draw cargo
        cargo_x = []
        cargo_y = []
        for j in range(-1, len(self.cargo)):
            cargo_x.append(self.cargo[j][0])
            cargo_y.append(self.cargo[j][1])
        self.draw_cargo, = self.ax.plot(cargo_x, cargo_y, color='k')
        # draw obstacle
        for i in range(len(self.obstacles)):
            self.ax.add_patch(plt.Circle(self.obstacles[i], self.obstacles_r[i], color='k'))
        # draw destination
        self.ax.plot(float(self.destination[0]), float(self.destination[1]), marker='x', color='k')

        plt.ion()
        plt.show()

        # add dynamic constraints
        self.CONSTRAINTS_DYNAMIC = []
        self.LB_DYNAMIC = []
        self.UB_DYNAMIC = []
        for j in range(self.ROBOT_NUM):
            for i in range(self.N - 1):
                # dynamic constraints
                v = self.u[2 * j + 0, i]
                w = self.u[2 * j + 1, i]
                theta = self.x[3 * j + 2, i]
                self.CONSTRAINTS_DYNAMIC.append(
                    self.x[3 * j + 0, i + 1] - (self.x[3 * j + 0, i] + v * cs.cos(theta) * self.dt))  # dx
                self.LB_DYNAMIC.append(0)
                self.UB_DYNAMIC.append(0)
                self.CONSTRAINTS_DYNAMIC.append(
                    self.x[3 * j + 1, i + 1] - (self.x[3 * j + 1, i] + v * cs.sin(theta) * self.dt))  # dy
                self.LB_DYNAMIC.append(0)
                self.UB_DYNAMIC.append(0)
                self.CONSTRAINTS_DYNAMIC.append(
                    self.x[3 * j + 2, i + 1] - (self.x[3 * j + 2, i] + w * self.dt))  # d\theta
                self.LB_DYNAMIC.append(0)
                self.UB_DYNAMIC.append(0)

                # beta constraints
                if i != self.N - 2:
                    self.CONSTRAINTS_DYNAMIC.append(self.u[2 * j + 1, i + 1] - self.u[2 * j + 1, i])
                    self.LB_DYNAMIC.append(-self.BETA_MAX)
                    self.UB_DYNAMIC.append(self.BETA_MAX)

                # velocity constraints
                self.CONSTRAINTS_DYNAMIC.append(self.u[2 * j + 0, i])
                self.LB_DYNAMIC.append(-self.VMAX)
                self.UB_DYNAMIC.append(self.VMAX)
                self.CONSTRAINTS_DYNAMIC.append(self.u[2 * j + 1, i])
                self.LB_DYNAMIC.append(-self.WMAX)
                self.UB_DYNAMIC.append(self.WMAX)

    def initialization(self):
        self.constraints = self.CONSTRAINTS_DYNAMIC[:]
        self.lb = self.LB_DYNAMIC[:]
        self.ub = self.UB_DYNAMIC[:]
        for j in range(3 * self.ROBOT_NUM):  # initial states
            self.add_constraints(self.x[j, 0] - self.x_current[j], 0, 0)

        # if self.phase == 'transporting':
        #     # formation constraints
        #     err_form = 0
        #     for i in range(self.N):
        #         err_form += self.cal_err_form(self.x[:, i])
        #     self.constraints.append(err_form)
        #     self.lb.append(0)
        #     self.ub.append(self.EPSILON_FORM)
        #
        #     # avoidance constraints
        #     for i in range(self.N):
        #         for k in range(len(self.obstacles)):
        #             self.constraints.append(self.cal_err_avoid(self.x[:, i], self.obstacles[k]))
        #             self.lb.append((self.obstacles_r[k] + self.cargo_r) ** 2)
        #             self.ub.append(cs.inf)

    def solve(self, cost, constraints, lb, ub, x0=None):
        X = cs.reshape(self.x, -1, 1)
        U = cs.reshape(self.u, -1, 1)

        # construct initial guess
        if x0 is None:
            x0 = [float(self.x_current[j]) for j in range(3 * self.ROBOT_NUM)] * self.N
            xc = self.cal_xc(self.x_current)
            coordinate_diff = self.destination - xc
            orientation = cs.arctan2(coordinate_diff[1], coordinate_diff[0])
            w_init = []
            for j in range(self.ROBOT_NUM):
                w_init.append(0)
                w_init.append(float((orientation - self.x_current[3 * j + 2]) * self.WMAX / cs.pi))
            x0 += w_init * (self.N - 1)

        nlp = {'x': cs.vertcat(X, U), 'f': cost, 'g': cs.vertcat(*constraints)}
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        # S = cs.nlpsol('S', 'ipopt', nlp)
        S = cs.nlpsol('S', 'ipopt', nlp, opts)
        solution = S(x0=x0, lbg=lb, ubg=ub)
        return solution

    def cal_err_form(self, states):
        df_index = 0
        err_form = 0
        for j in range(self.ROBOT_NUM - 1):
            for k in range(j + 1, self.ROBOT_NUM):
                err_form += (cs.sumsqr(states[3 * j: 3 * j + 2] - states[3 * k: 3 * k + 2])
                             - self.df_star[df_index] ** 2) ** 2
                df_index += 1
        return err_form

    def cal_xc(self, states):
        xc = cs.SX([0, 0])
        for j in range(self.ROBOT_NUM):
            xc[0] += states[3 * j + 0]
            xc[1] += states[3 * j + 1]
        xc /= self.ROBOT_NUM

        return xc

    def cal_err_avoid(self, states, obs, obs_r=0, v_obs=False):
        xc = self.cal_xc(states)
        if v_obs:
            err_avoid = cs.sumsqr(xc - obs) ** 2
        else:
            err_avoid = (cs.sumsqr(xc - obs) - (self.cargo_r + self.WIDTH + obs_r) ** 2) ** 2

        if self.BARRIER_FUNCTION == 'log':
            err_avoid = - cs.log(err_avoid) / 5
        elif self.BARRIER_FUNCTION == 'inverse':
            err_avoid = 1 / err_avoid
        else:
            err_avoid = 1 / err_avoid

        return err_avoid

    def moveto_cargo(self):  # robots move to corresponding vertices of cargo without collision avoidance
        cost_navi = 0
        for i in range(self.N):
            for j in range(self.ROBOT_NUM):
                cost_navi += cs.sumsqr(self.x[3 * j: 3 * j + 2, i] - self.cargo[j])

        x0 = [float(self.x_current[j]) for j in range(3 * self.ROBOT_NUM)] * self.N
        x0 += [0] * 2 * self.ROBOT_NUM * (self.N - 1)

        solution = self.solve(cost_navi, self.constraints, self.lb, self.ub, x0)
        self.u_current = solution['x'][3 * self.ROBOT_NUM * self.N: 3 * self.ROBOT_NUM * self.N + 2 * self.ROBOT_NUM]

    def formation(self):
        cost_form = 0
        for i in range(self.N):
            cost_form += self.cal_err_form(self.x[:, i])
        solution = self.solve(cost_form, self.constraints, self.lb, self.ub)
        self.err_form = float(solution['f'])
        self.add_constraints(cost_form, self.err_form, self.err_form + self.EPSILON_FORM)
        self.u_current = solution['x'][3 * self.ROBOT_NUM * self.N: 3 * self.ROBOT_NUM * self.N + 2 * self.ROBOT_NUM]

    def avoidance(self):
        cost_avoid = 0
        for i in range(self.N):
            for k in range(len(self.obstacles)):
                cost_avoid += self.cal_err_avoid(self.x[:, i], self.obstacles[k],
                                                 self.obstacles_r[k])

        for k in range(len(self.virtual_obstacles)):
            for i in range(self.N):
                cost_avoid += self.cal_err_avoid(self.x[:, i], self.virtual_obstacles[k], v_obs=True)

        solution = self.solve(cost_avoid, self.constraints, self.lb, self.ub)
        self.err_avoid = float(solution['f'])
        self.add_constraints(cost_avoid, self.err_avoid, self.err_avoid + self.EPSILON_AVOID)
        self.u_current = solution['x'][3 * self.ROBOT_NUM * self.N: 3 * self.ROBOT_NUM * self.N + 2 * self.ROBOT_NUM]

    def navigation(self):
        xc = self.cal_xc(self.x[:, self.N - 1])
        cost_navi = cs.sumsqr(xc - self.destination)
        # for i in range(self.N):
        #     xc = self.cal_xc(self.x[:, i])
        #     cost_navi += cs.sumsqr(xc - self.destination)

        solution = self.solve(cost_navi, self.constraints, self.lb, self.ub)
        self.err_navi = float(solution['f'])
        self.add_constraints(cost_navi, self.err_navi, self.err_navi + self.EPSILON_NAVI)
        self.u_current = solution['x'][3 * self.ROBOT_NUM * self.N: 3 * self.ROBOT_NUM * self.N + 2 * self.ROBOT_NUM]
        # print("navigating")

        # if stuck in obstacle
        # xN = solution['x'][3 * self.ROBOT_NUM * (self.N - 1): 3 * self.ROBOT_NUM * self.N]
        # err = cs.sumsqr(self.xN - xN)
        # if err < 1:
        #     self.it_stuck += 1
        # else:
        #     self.it_stuck = 0
        # if self.it_stuck > 10:
        #     print('stuck')
        #     xc = self.cal_xc(xN)
        #     self.add_virtual_obstacle(xc)
        #     self.it_stuck = 0
        # self.xN = xN

    def delta_u(self):
        cost_du = 0
        for i in range(self.N - 2):
            for j in range(self.ROBOT_NUM):
                v   = self.u[2 * j + 0, i]
                v_n = self.u[2 * j + 0, i + 1]
                w   = self.u[2 * j + 1, i]
                cost_du += w ** 2 + (v - v_n)

        solution = self.solve(cost_du, self.constraints, self.lb, self.ub)
        self.u_current = solution['x'][3 * self.ROBOT_NUM * self.N: 3 * self.ROBOT_NUM * self.N + 2 * self.ROBOT_NUM]

    def update_state(self):
        # xc = self.cal_xc(self.x_current)
        # distance = float(cs.norm_2(xc - self.c_obs))
        # self.TOLERANCE_AVOID = 5e-3 * distance * self.N
        for j in range(self.ROBOT_NUM):
            v = self.u_current[2 * j + 0]
            w = self.u_current[2 * j + 1]
            theta = self.x_current[3 * j + 2]

            self.x_current[3 * j + 0] = self.x_current[3 * j + 0] + v * cs.cos(theta) * self.dt
            self.x_current[3 * j + 1] = self.x_current[3 * j + 1] + v * cs.sin(theta) * self.dt
            self.x_current[3 * j + 2] = self.x_current[3 * j + 2] + w * self.dt

    def add_constraints(self, constraint, lb, ub):
        self.constraints.append(constraint)
        self.lb.append(lb)
        self.ub.append(ub)

    def add_virtual_obstacle(self, position):
        self.virtual_obstacles.append(position)

    def compute_vertices(self, x, y, theta):
        phi1 = np.pi / 2 - np.arctan(self.LENGTH / self.WIDTH)
        phi2 = np.pi - phi1
        B = np.linalg.norm([self.LENGTH, self.WIDTH]) / 2
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

    def draw(self):
        # draw robot and formation
        cargo_x = [self.x_current[-3]]
        cargo_y = [self.x_current[-2]]
        for j in range(self.ROBOT_NUM):
            self.traj[2 * j + 0].append(float(self.x_current[3 * j + 0]))
            self.traj[2 * j + 1].append(float(self.x_current[3 * j + 1]))
            # casadi.SX typed output cannot be used in plt
            x = float(self.x_current[3 * j + 0])
            y = float(self.x_current[3 * j + 1])
            theta = float(self.x_current[3 * j + 2])
            vertices = self.compute_vertices(x, y, theta)
            self.draw_robot[j].set_data(vertices[0], vertices[1])
            self.draw_traj[j].set_data(self.traj[2 * j + 0], self.traj[2 * j + 1])

            cargo_x.append(x)
            cargo_y.append(y)

        # if self.phase == 'navigating':
        #     self.draw_cargo.set_data(cargo_x, cargo_y)

        plt.pause(0.01)


if __name__ == '__main__':
    map = cv.imread('maps/map2.png')
    hmpc = HMPC(map)
    it = 0

    while hmpc.cal_err_form(hmpc.x_current) > hmpc.EPSILON_FORM:
        hmpc.initialization()
        if hmpc.phase == 'moving_to_cargo':
            hmpc.moveto_cargo()
            print('iteration: {}, moving to cargo'.format(it))
        hmpc.update_state()
        hmpc.draw()
        it += 1

    hmpc.phase = 'transporting'
    while True:
        hmpc.initialization()
        if hmpc.phase == 'transporting':
            hmpc.formation()
            hmpc.avoidance()
            hmpc.navigation()
            # hmpc.delta_u()

            # xc = hmpc.cal_xc(hmpc.x_current)
            # dis_obs = cs.norm_2(xc - hmpc.c_obs)
            # print('distance: ', dis_obs)
            print('iteration: {}, navigating'.format(it))
        elif hmpc.phase == 'forming':
            hmpc.formation()
            print('iteration: {}, forming'.format(it))

        if hmpc.cal_err_form(hmpc.x_current) > hmpc.EPSILON_FORM:
            hmpc.phase = 'forming'
        else:
            hmpc.phase = 'transporting'
        hmpc.update_state()
        hmpc.draw()
        it += 1
