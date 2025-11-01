"""
广度优先搜索算法 - 2D路径规划
@author: Ethan.Geng
"""

import os
import sys
from collections import deque

# 将搜索模块路径添加到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

from Search_2D import plotting, env
from Search_2D.Astar import AStar
import math
import heapq

class BFS(AStar):
    """
    BFS 广度优先搜索类
    继承自 AStar 类，通过修改节点优先级实现 BFS
    BFS 将新访问的节点添加到 openset 的末尾，确保按层级顺序搜索
    标准 BFS：只允许上下左右4个方向移动
    """
    def __init__(self, s_start, s_goal, heuristic_type):
        """
        初始化 BFS
        :param s_start: 起点
        :param s_goal: 终点
        :param heuristic_type: 启发式类型（BFS不使用，但保留接口兼容性）
        """
        super().__init__(s_start, s_goal, heuristic_type)
        # 重写移动方向：只允许上下左右4个方向
        # (x, y): 上(0,1), 下(0,-1), 左(-1,0), 右(1,0)
        self.u_set = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    
    def cost(self, s_start, s_goal):
        """
        重写代价函数：BFS 中所有移动的代价都为 1
        这样才是真正的广度优先搜索（按层扩展）
        :param s_start: 起始节点
        :param s_goal: 目标节点
        :return: 移动代价（恒为1）
        """
        # 检查碰撞
        if self.is_collision(s_start, s_goal):
            return math.inf
        
        # BFS 核心：所有移动代价都是 1（只允许上下左右移动）
        return 1
    
    def searching(self):
        """
        广度优先搜索算法主函数
        :return: path (路径列表), visited (访问顺序列表)
        """

        # 初始化起点：父节点指向自己
        self.PARENT[self.s_start] = self.s_start
        # 起点的代价为 0
        self.g[self.s_start] = 0
        # 终点的初始代价设为无穷大
        self.g[self.s_goal] = math.inf
        # 将起点加入 OPEN 列表（优先级为 0）
        heapq.heappush(self.OPEN,
                       (0, self.s_start))

        # 主循环：当 OPEN 列表不为空时继续搜索
        while self.OPEN:
            # 从 OPEN 列表中弹出优先级最小的节点（FIFO 队列行为）
            _, s = heapq.heappop(self.OPEN)
            # 将当前节点加入 CLOSED 列表（已访问）
            self.CLOSED.append(s)

            # 如果到达目标点，停止搜索
            if s == self.s_goal:
                break

            # 遍历当前节点的所有邻居节点
            for s_n in self.get_neighbor(s):
                # 计算从起点经过当前节点到邻居节点的代价
                new_cost = self.g[s] + self.cost(s, s_n)

                # 如果邻居节点未被访问过，初始化其代价为无穷大
                if s_n not in self.g:
                    self.g[s_n] = math.inf

                # 如果找到更优路径，更新节点信息
                if new_cost < self.g[s_n]:
                    # 更新邻居节点的代价
                    self.g[s_n] = new_cost
                    # 记录邻居节点的父节点
                    self.PARENT[s_n] = s

                    # BFS 核心：将新节点添加到 openset 的末尾
                    # 通过递增优先级值确保 FIFO（先进先出）队列行为
                    prior = self.OPEN[-1][0]+1 if len(self.OPEN)>0 else 0
                    heapq.heappush(self.OPEN, (prior, s_n))

        # 返回路径和访问顺序
        return self.extract_path(self.PARENT), self.CLOSED


def main():
    """
    主函数：演示 BFS 算法的使用
    """
    # 定义起点坐标
    s_start = (5, 5)
    # 定义终点坐标
    s_goal = (45, 25)

    # 创建 BFS 搜索对象
    # 参数：起点、终点、启发式函数类型（'None' 表示不使用启发式）
    bfs = BFS(s_start, s_goal, 'None')
    # 创建绘图对象，用于可视化搜索过程
    plot = plotting.Plotting(s_start, s_goal)

    # 执行 BFS 搜索，获取路径和访问顺序
    path, visited = bfs.searching()
    # 动画展示搜索过程和最终路径
    # 传递 bfs.g 代价字典，用于显示每个节点的距离标注
    plot.animation(path, visited, "Breadth-first Searching (BFS)", cost_dict=bfs.g)


if __name__ == '__main__':
    main()
