import numpy as np
import matplotlib.pyplot as plt


def cal_xc(robot_num, states):
    '''
    :param states: x_current
    :return:
    '''
    xc = np.array([0, 0], dtype=np.float32)
    for j in range(robot_num):
        xc[0] += states[2 * j + 0]
        xc[1] += states[2 * j + 1]
    xc = xc / robot_num

    return xc


def cal_xh(robot_num, a, x_current):
    """
    calculate the reference point for the orientation MPC problem
    :param x_current:
    :return:
    """
    xh = np.array([0, 0], dtype=np.float32)
    for i in range(robot_num):
        xh[0] += a[i] * x_current[2 * i + 0]  # x
        xh[1] += a[i] * x_current[2 * i + 1]  # y

    return xh


def cal_theta(xc, xh):
    """
    calculate the orientation of the formation
    """
    theta = np.arctan2(xh[1] - xc[1], xh[0] - xc[0])
    return theta


def draw(data, time, xlabel, ylabel, colors, fontsize, path, SAVE, label='Centroid', ylim=None):
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([0, time[-1] + 1])
    if ylim is not None:
        plt.ylim(ylim)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    drawer = plt.gca()
    plt.sca(drawer)
    if len(data.shape) == 1:
        drawer.plot(time, data, label=label)
    else:
        for i in range(data.shape[0]):
            drawer.plot(time, data[i], color=colors[i], label='Vehicle ' + str(i + 1))
    plt.legend(fontsize=fontsize)
    if SAVE:
        plt.savefig(path, bbox_inches="tight")
        plt.savefig(path, bbox_inches="tight")
    else:
        plt.show()


SAVE = True
map_name = 'dyna_obs2'
save_path = '../output/' + map_name
colors = ['tomato', 'orange', 'royalblue', 'darkgreen']

res = np.load(save_path + '/res.npz')
# print(res.files)

its = res['iterations']
dt = res['dt']
length = res['length']
omega = res['omega']
df_star = res['df_star'] / length
traj = res['traj'] / length  # of shape [8, its]
formation_error = res['formation_error'] / length
velocity = res['velocity'] / length
robot_num = int(traj.shape[0] / 2)

a = [0, 0.5, 0.5, 0]
time = np.linspace(0, its, its) * dt

x_c = cal_xc(robot_num, traj[:, 0])
x_h = cal_xh(robot_num, a, traj[:, 0])
theta_c = cal_theta(x_c, x_h)
for i in range(1, its):
    x_c_tmp = cal_xc(robot_num, traj[:, i])
    x_h_tmp = cal_xh(robot_num, a, traj[:, i])
    theta_c_tmp = cal_theta(x_c_tmp, x_h_tmp)
    x_c = np.vstack((x_c, x_c_tmp))
    x_h = np.vstack((x_h, x_h_tmp))
    theta_c = np.append(theta_c, theta_c_tmp)

velocity_c = np.array([], dtype=np.float32)
omega_c = np.array([], dtype=np.float32)
for i in range(its - 1):
    velocity_c = np.append(velocity_c, np.linalg.norm(x_c[i + 1] - x_c[i]) / dt)
    omega_c = np.append(omega_c, (theta_c[i+1] - theta_c[i]) / dt)

fontsize = 22
# plt.rc('text', usetex=True)  # 启用 LaTeX 渲染
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']


plt.figure()
plt.xlabel('t (s)', fontsize=fontsize)
plt.ylabel('Formation error (m)', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
form_err_drawer = plt.gca()
plt.sca(form_err_drawer)

# for i in range(formation_error.shape[0]):
#     form_err_drawer.plot(time, formation_error[i])
k = 0
plt.xlim([0, time[-1] + 3])
while k < formation_error.shape[0]:
    for i in range(robot_num - 1):
        for j in range(i + 1, robot_num):
            # form_err_drawer.plot(index2plot, formation_error[k][index2plot], label='$e_{' + str(i+1) + ',' + str(j+1) + '}$')
            form_err_drawer.plot(time, formation_error[k], label='e$_{' + str(i+1) + ',' + str(j+1) + '}$')
            k += 1

print(np.average(np.fabs(formation_error)))
form_err_drawer.yaxis.get_offset_text().set_fontsize(fontsize)

plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=fontsize, fancybox=True)
# plt.legend()
if SAVE:
    plt.savefig(save_path + '/formation_error.svg', bbox_inches="tight")
    plt.savefig(save_path + '/formation_error.png', bbox_inches="tight")
    np.savetxt(save_path + '/mean_form_err.txt', [np.average(np.fabs(formation_error)) / len(df_star), np.max(np.fabs(formation_error)), np.max(df_star)])
else:
    plt.show()


# # draw velocity
# draw(velocity, time, 't (s)', 'Velocity (m/s)', colors, fontsize, save_path + '/velocity.svg', SAVE, ylim=[0, 0.65])
# # draw angular velocity
# draw(omega, time, 't (s)', 'Angular velocity (rad/s)', colors, fontsize, save_path + '/omega.svg', SAVE, ylim=[-1.05, 1.05])
# # draw orientation of the formation
# draw(theta_c, time, 't (s)', 'Orientation (rad)', colors, fontsize, save_path + '/theta_c.svg', SAVE, ylim=[0.6, 1.6])
# time = np.delete(time, -1)
# # draw velocity of the centroid
# draw(velocity_c, time, 't (s)', 'Velocity (m/s)', colors, fontsize, save_path + '/velocity_c.svg', SAVE, ylim=[0, 0.65])
# # draw angular velocity of the centroid
# draw(omega_c, time, 't (s)', 'Angular velocity (rad/s)', colors, fontsize, save_path + '/omega_c.svg', SAVE, ylim=[-0.2, 0.1])
