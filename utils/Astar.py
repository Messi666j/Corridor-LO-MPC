import cv2 as cv
from utils.VisibilityGraph import getVisGraph
import math as m


class astar:
    # (p_i, p_f, obss, vis_graph):
    def __init__(self, p_i, p_f, obss):
        self.p_i = p_i
        self.p_f = p_f
        self.vis_graph = getVisGraph(p_i, p_f, obss)
        # self.vis_graph = vis_graph
        self.obss = obss
        self.obss.append([[p_i[0], p_i[1]]])
        self.obss.append([[p_f[0], p_f[1]]])
        self.open_set = []
        self.close_set = []
        self.path_index = []
        self.path_position = []
        self.g_p = 0
        self.path_found = False
        self.path_found_index = 0

    class Node:
        def __init__(self, index, g, h, neighbour_index, parent):
            """
            :param index: 顶点的索引，格式为[障碍物索引，顶点在障碍物上的索引]
            :param f: 代价函数
            :param neighbour: 可视的顶点的索引
            :param parent: 父节点
            """
            self.index = index
            self.g = g
            self.h = h
            self.f = 0
            self.neighbour_index = neighbour_index
            self.parent = parent

    def calDistance(self, p1, p2):
        # return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        return m.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def calH(self, p, p_f):
        return self.calDistance(p, p_f)

    def calG(self, g_p, p, p_n):
        return g_p + self.calDistance(p, p_n)

    def searchF(self, array, target):
        if len(array) == 0:
            return 0
        low = 0
        high = len(array) - 1
        while low <= high:
            mid = (low + high) // 2
            if array[mid].f > target:
                high = mid - 1
            elif array[mid].f < target:
                low = mid + 1
            else:
                return mid
        return mid if array[mid].f > target else mid + 1

    def searchPath(self):
        '''
        :return: index: 顶点索引 position: 顶点坐标
        '''
        # 初始化
        nodes = [0] * (len(self.obss) - 2)
        for i in range(len(nodes)):
            nodes[i] = [0] * len(self.obss[i])
            for j in range(len(self.obss[i])):
                nodes[i][j] = self.Node([i, j], None, self.calH(self.obss[i][j], self.p_f), self.vis_graph[i][j], None)
        start_node = self.Node([-2, 0], 0, self.calH(self.p_i, self.p_f), self.vis_graph[-2][0], None)
        end_node = self.Node([-1, 0], None, 0, self.vis_graph[-1][0], None)
        nodes.append([start_node])
        nodes.append([end_node])

        self.open_set.append(start_node)

        while len(self.open_set) > 0:
            current_node = self.open_set[0]  # open_set是以f从小到大排序，open_set[0]是f最小的点

            if current_node.index == [-1, 0]:
                print("find path")
                break

            self.open_set.remove(current_node)
            self.close_set.append(current_node)

            neighbours_index = current_node.neighbour_index
            for ni_x, ni_y in neighbours_index:
                if [ni_x, ni_y] in [node.index for node in self.close_set]:  # 如果已经在close_set中，则忽略
                    continue

                # 当前节点到邻居节点的g值
                new_g = current_node.g + \
                        self.calDistance(self.obss[current_node.index[0]][current_node.index[1]], self.obss[ni_x][ni_y])
                new_f = new_g + nodes[ni_x][ni_y].h
                isIn = False
                for open_node in self.open_set:
                    if [ni_x, ni_y] == open_node.index:  # 如果在open_set中
                        isIn = True
                        if new_f < open_node.f:  # 如果新的f是最优，更新open_set
                            self.open_set.remove(open_node)
                            nodes[ni_x][ni_y].g = new_g
                            nodes[ni_x][ni_y].f = new_f
                            nodes[ni_x][ni_y].parent = current_node.index
                            f_index = self.searchF(self.open_set, nodes[ni_x][ni_y].f)
                            self.open_set.insert(f_index, nodes[ni_x][ni_y])
                        break
                if not isIn:  # 如果不在open_set中
                    nodes[ni_x][ni_y].g = new_g
                    nodes[ni_x][ni_y].f = new_f
                    nodes[ni_x][ni_y].parent = current_node.index
                    f_index = self.searchF(self.open_set, nodes[ni_x][ni_y].f)
                    self.open_set.insert(f_index, nodes[ni_x][ni_y])

        if len(self.open_set) == 0:
            print("no path")
        else:
            n = end_node
            while n.index != start_node.index:
                self.path_index.insert(0, n.index)
                n = nodes[n.parent[0]][n.parent[1]]
            self.path_index.insert(0, start_node.index)

            self.path_position = [0] * len(self.path_index)

            for i in range(len(self.path_index)):
                self.path_position[i] = self.obss[self.path_index[i][0]][self.path_index[i][1]]

            return self.path_index, self.path_position


if __name__ == '__main__':
    # pass
    DISP = True
    SAVE = True
    p_i = [500, 400]  # 起点位置，索引固定为[-2, 0]
    p_f = [520, 500]  # 终点位置，索引固定为[-1, 0]
    img_path = 'obstacles.png'
    img = cv.imread(img_path)
    # img = cv.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
    obss = obstacle_detect(img)
    vis_graph = getVisGraph(p_i, p_f, obss)

    # obss.append([[p_i[0], p_i[1]]])
    # obss.append([[p_f[0], p_f[1]]])

    a = astar(p_i, p_f, obss)
    index, position = a.searchPath()
    cv.circle(img, p_i,  5, (0, 0, 255), -1)
    cv.circle(img, p_f, 5, (0, 0, 255), -1)
    for i in range(len(vis_graph)):  # 画可视图
        for j in range(len(vis_graph[i])):
            for k in range(len(vis_graph[i][j])):
                cv.line(img, obss[vis_graph[i][j][k][0]][vis_graph[i][j][k][1]], obss[i][j], (255, 0, 0), 1)

    for i in range(len(position) - 1):  # 画最优路径
        cv.line(img, position[i], position[i+1], (0, 0, 255), 1)

    cv.imshow('img', img)
    cv.waitKey(0)
    # cv.imwrite('optimal path.png', img)

