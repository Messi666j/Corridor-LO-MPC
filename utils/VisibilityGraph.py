import math


def crossProduct(p11, p12, p21, p22):  # 叉积
    # print(p11, p12, p21, p22)
    vec1 = [p11[0] - p12[0], p11[1] - p12[1]]
    vec2 = [p21[0] - p22[0], p21[1] - p22[1]]
    product = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    return product


def isTangent(p, v_i, obs):
    # https://blog.csdn.net/u013279723/article/details/104239723
    """
    :param p: 多边形外点
    :param v_i: 多边形上待判断顶点的索引
    :param obs: 障碍物所有点
    :return:
    """
    v = obs[v_i]
    if v_i == 0:
        v_p = obs[-1]
        v_n = obs[1]
    elif v_i == len(obs) - 1:
        v_p = obs[v_i - 1]
        v_n = obs[0]
    else:
        v_p = obs[v_i - 1]
        v_n = obs[v_i + 1]

    p1 = crossProduct(p, v, v_p, v)  # 叉积1
    p2 = crossProduct(p, v, v, v_n)  # 叉积2

    if (p1 >= 0 and p2 >= 0) or (p1 <= 0 and p2 <= 0):
        return False
    else:
        return True


def isIntersect(p11, p12, p21, p22):
    # https://blog.csdn.net/zhouzi2018/article/details/81871875
    if (abs(p11[0] - p12[0]) + abs(p21[0] - p22[0]) <
    max([p11[0], p12[0], p21[0], p22[0]]) - min([p11[0], p12[0], p21[0], p22[0]])) or \
    (abs(p11[1] - p12[1]) + abs(p21[1] - p22[1]) <
    max([p11[1], p12[1], p21[1], p22[1]]) - min([p11[1], p12[1], p21[1], p22[1]])):
        return False
    else:
        p1 = crossProduct(p11, p22, p21, p22)
        p2 = crossProduct(p12, p22, p21, p22)
        p3 = crossProduct(p21, p12, p11, p12)
        p4 = crossProduct(p22, p12, p11, p12)
        if (p1 >= 0 and p2 >= 0) or (p1 <= 0 and p2 <= 0) or (p3 >= 0 and p4 >= 0) or (p3 <= 0 and p4 <= 0):
            return False
        else:
            return True


def calAngle(p1, p2):
    angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    return angle if angle >= 0 else angle + 2 * 3.1416


class RelativeAngle:  # 用于排序
    def __init__(self, o_i, v_i, angle):
        """
        :param o_i: 障碍物索引
        :param v_i: 障碍物顶点索引
        :param angle: 相对角度
        """
        self.o_i = o_i
        self.v_i = v_i
        self.angle = angle

    def __str__(self):  # 自定义打印类的内容
        return "{}, {}, {}".format(self.o_i, self.v_i, self.angle)


def cmp(self, other):
    if self.angle < other.angle:
        return 1
    elif self.angle == other.angle:
        return 0
    else:
        return -1


def rpsa(v, obss, index, shape):
    """
    :param v: 待计算可视点的顶点
    :param obss: 所有障碍物
    :param index: 当前顶点索引

    :return: 顶点v的所有可视顶点索引
    """
    # https://fribbels.github.io/shortestpath/writeup.html
    v_tmp = [shape, v[1]]  # 初始线段的右端点
    S = []
    angle_list = []
    visible_list = []
    if index is None:
        visible_list = []
    elif index[1] == len(obss[index[0]]) - 1:
        visible_list = [[index[0], index[1] - 1], [index[0], 0]]
    else:
        visible_list = [[index[0], index[1] + 1], [index[0], index[1] - 1]]
    for i in range(len(obss)):
        if index is not None:
            if i == index[0]:  # 跳过顶点v自身所在的障碍物
                continue
        for j in range(len(obss[i])):
            angle_list.append(RelativeAngle(i, j, calAngle(v, obss[i][j])))  # 计算每个顶点相对当前点的角度
            if isIntersect(v, v_tmp, obss[i][j-1], obss[i][j]):
                S.append([obss[i][j-1], obss[i][j]])  # 计算初始S与哪些边相交
    sorted_vertices = sorted(angle_list, key=lambda x: x.angle)

    for vertex in sorted_vertices:  # 按角度顺序遍历每一个点
        isVisible = True
        if index is None:  # 起点或终点不需要判断相互相切
            isTan = isTangent(v, vertex.v_i, obss[vertex.o_i])
        else:  # 如果两个点互为切点
            isTan = (isTangent(v, vertex.v_i, obss[vertex.o_i]) and
                     isTangent(obss[vertex.o_i][vertex.v_i], index[1], obss[index[0]]))
        if isTan:
            for s in S:  # 遍历S中的每一条线段
                if isIntersect(s[0], s[1], v, obss[vertex.o_i][vertex.v_i]):
                    isVisible = False
                    break
        else:
            isVisible = False

        if isVisible:
            visible_list.append([vertex.o_i, vertex.v_i])

        # 更新S
        try:
            del S[S.index([obss[vertex.o_i][vertex.v_i - 1], obss[vertex.o_i][vertex.v_i]])]
        except:
            S.append([obss[vertex.o_i][vertex.v_i - 1], obss[vertex.o_i][vertex.v_i]])
        if vertex.v_i == len(obss[vertex.o_i]) - 1:
            try:
                del S[S.index([obss[vertex.o_i][vertex.v_i], obss[vertex.o_i][0]])]
            except:
                S.append([obss[vertex.o_i][vertex.v_i], obss[vertex.o_i][0]])
        else:
            try:
                del S[S.index([obss[vertex.o_i][vertex.v_i], obss[vertex.o_i][vertex.v_i + 1]])]
            except:
                S.append([obss[vertex.o_i][vertex.v_i], obss[vertex.o_i][vertex.v_i + 1]])

    return visible_list


def getVisGraph(p_i, p_f, obss):
    """
    :param p_i: 起点
    :param p_f: 终点
    :param img: 图
    :param disp: 是否显示可视图
    :return: 可视图visGraph，四维数组
    第一维是障碍物索引，第二维是顶点索引，第三维是该顶点的所有可视顶点，第四维是可视顶点的索引
    起点与终点在-2和-1的位置
    """
    visGraph = []
    most_right = 0
    for obs in obss:
        for vertex in obs:
            if vertex[0] > most_right:
                most_right = vertex[0]
    most_right += 2
    for i in range(len(obss)):
        visVertex = []
        for j in range(len(obss[i])):
            visList = rpsa(obss[i][j], obss, [i, j], most_right)
            visVertex.append(visList)
        visGraph.append(visVertex)

    i_connect_f = True
    for i in range(len(obss)):
        for j in range(len(obss[i])):
            if isIntersect(p_i, p_f, obss[i][j - 1], obss[i][j]):
                i_connect_f = False
                break
        if not i_connect_f:
            break

    visVertex = []
    visList = rpsa(p_i, obss, None, most_right)  # 起点
    for o, v in visList:
        visGraph[o][v].append([-2, 0])
    if i_connect_f:
        visList.append([-1, 0])
    visVertex.append(visList)
    visGraph.append(visVertex)

    visVertex = []
    visList = rpsa(p_f, obss, None, most_right)  # 终点
    for o, v in visList:
        visGraph[o][v].append([-1, 0])
    if i_connect_f:
        visList.append([-2, 0])
    visVertex.append(visList)
    visGraph.append(visVertex)

    return visGraph


if __name__ == '__main__':
    pass
    # p_i = [20, 20]  # 起点横纵坐标和朝向
    # p_f = [800, 400]  # 终点横纵坐标和朝向
    # img = cv.imread('obstacles.png')
    # img = cv.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
    #
    # obss = obstacle_detect(img)
    #
    # getVisGraph(p_i, p_f, obss, disp=True)
