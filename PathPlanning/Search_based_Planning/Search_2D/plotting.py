"""
Plot tools 2D
2D 路径规划可视化工具
用于绘制搜索过程、路径和障碍物的动画展示
@author: huiming zhou
"""

import os
import sys
import matplotlib.pyplot as plt

# 将搜索模块路径添加到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

from Search_2D import env


class Plotting:
    """
    绘图类：用于可视化 2D 路径规划算法的搜索过程和结果
    """
    def __init__(self, xI, xG):
        """
        初始化绘图对象
        :param xI: 起点坐标 (x, y)
        :param xG: 终点坐标 (x, y)
        """
        self.xI, self.xG = xI, xG
        # 创建环境对象
        self.env = env.Env()
        # 获取障碍物地图
        self.obs = self.env.obs_map()

    def update_obs(self, obs):
        """
        更新障碍物地图
        :param obs: 新的障碍物列表
        """
        self.obs = obs

    def animation(self, path, visited, name, cost_dict=None, terrain_colors=None):
        """
        标准动画函数：展示搜索过程和最终路径
        适用于大多数搜索算法（BFS, DFS, A*, Dijkstra等）
        :param path: 最终路径节点列表
        :param visited: 已访问节点列表（按访问顺序）
        :param name: 算法名称（显示在标题）
        :param cost_dict: 可选，节点代价字典，用于显示距离标注
        :param terrain_colors: 可选，地形颜色字典 {节点: 颜色}
        """
        self.plot_grid(name)                    # 绘制网格和障碍物
        
        # 先绘制地形节点（作为背景）
        if terrain_colors:
            self.plot_terrain(terrain_colors)
        
        self.plot_visited(visited, cost_dict=cost_dict)  # 绘制访问过程（动画）
        self.plot_path(path)                    # 绘制最终路径
        plt.show()                              # 显示图形窗口

    def animation_lrta(self, path, visited, name):
        """
        LRTA* 算法专用动画函数
        LRTA* 是一种实时启发式搜索算法，会多次重新规划路径
        :param path: 路径列表的列表（每次重规划的路径）
        :param visited: 访问节点列表的列表（每次重规划的访问节点）
        :param name: 算法名称
        """
        self.plot_grid(name)
        cl = self.color_list_2()  # 获取颜色列表，用于区分不同迭代
        path_combine = []  # 合并所有路径

        # 逐次显示每次重规划的过程
        for k in range(len(path)):
            self.plot_visited(visited[k], cl[k])  # 绘制当前迭代的访问节点
            plt.pause(0.2)
            self.plot_path(path[k])               # 绘制当前迭代的路径
            path_combine += path[k]
            plt.pause(0.2)
        # 移除起点（避免重复显示）
        if self.xI in path_combine:
            path_combine.remove(self.xI)
        self.plot_path(path_combine)  # 绘制完整路径
        plt.show()

    def animation_ara_star(self, path, visited, name):
        """
        ARA* 算法专用动画函数
        ARA* 是一种任意时间算法，会逐步优化路径（降低启发式权重）
        :param path: 路径列表的列表（每次优化的路径）
        :param visited: 访问节点列表的列表（每次优化的访问节点）
        :param name: 算法名称
        """
        self.plot_grid(name)
        cl_v, cl_p = self.color_list()  # 获取访问节点和路径的颜色列表

        # 逐次显示每次优化的过程
        for k in range(len(path)):
            self.plot_visited(visited[k], cl_v[k])  # 用不同颜色显示访问节点
            self.plot_path(path[k], cl_p[k], True)  # 用不同颜色显示路径
            plt.pause(0.5)  # 每次优化后暂停，便于观察

        plt.show()

    def animation_bi_astar(self, path, v_fore, v_back, name):
        """
        双向 A* 算法专用动画函数
        同时从起点和终点进行搜索，直到两个搜索树相遇
        :param path: 最终路径
        :param v_fore: 前向搜索访问的节点列表
        :param v_back: 后向搜索访问的节点列表
        :param name: 算法名称
        """
        self.plot_grid(name)
        self.plot_visited_bi(v_fore, v_back)  # 绘制双向搜索过程
        self.plot_path(path)                  # 绘制最终路径
        plt.show()

    def plot_grid(self, name):
        """
        绘制网格、起点、终点和障碍物
        :param name: 图形标题（通常是算法名称）
        """
        # 提取障碍物的 x 和 y 坐标
        obs_x = [x[0] for x in self.obs]
        obs_y = [x[1] for x in self.obs]

        # 绘制起点（蓝色方块）
        plt.plot(self.xI[0], self.xI[1], "bs", markersize=8)
        # 绘制终点（绿色方块）
        plt.plot(self.xG[0], self.xG[1], "gs", markersize=8)
        # 绘制障碍物（黑色方块）
        plt.plot(obs_x, obs_y, "sk")
        # 设置图形标题
        plt.title(name)
        # 设置坐标轴比例相等（确保网格是正方形）
        plt.axis("equal")
    
    def plot_terrain(self, terrain_colors):
        """
        绘制地形代价节点（彩色背景）
        :param terrain_colors: 地形颜色字典 {节点: 颜色}
        """
        for node, color in terrain_colors.items():
            # 绘制地形节点（大圆圈，作为背景）
            plt.plot(node[0], node[1], marker='o', color=color, 
                    markersize=15, alpha=0.6)

    def plot_visited(self, visited, cl='gray', cost_dict=None):
        """
        绘制已访问的节点（动画显示搜索过程）
        :param visited: 已访问节点列表
        :param cl: 节点颜色
        :param cost_dict: 可选，节点代价字典 {节点: 代价值}，用于显示距离标注
        """
        if self.xI in visited:
            visited.remove(self.xI)

        if self.xG in visited:
            visited.remove(self.xG)

        count = 0

        for x in visited:
            count += 1
            plt.plot(x[0], x[1], color=cl, marker='o', markersize=10)
            
            # ========== 距离标注显示 ==========
            # 如果提供了代价字典，在节点圆圈中心显示距离数字
            if cost_dict is not None and x in cost_dict:
                # 显示距离数字（整数显示，避免圆圈太小放不下）
                distance = cost_dict[x]
                plt.text(x[0], x[1], f'{int(distance)}', 
                        fontsize=7, color='white', weight='bold',
                        ha='center', va='center')
            
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])

            # ========== 动画速度控制参数 ==========
            # 原始设置：根据进度调整刷新频率（注释掉）
            # if count < len(visited) / 3:
            #     length = 20
            # elif count < len(visited) * 2 / 3:
            #     length = 30
            # else:
            #     length = 40
            
            # 更详细的显示：每个节点都暂停（可调整）
            # 调整建议：
            #   - length = 1  : 每个节点都显示（最详细，但慢）
            #   - length = 5  : 每5个节点刷新一次（较详细）
            #   - length = 10 : 每10个节点刷新一次（适中）
            length = 10  

            if count % length == 0:
                # 暂停时间调整建议：
                #   - 0.01秒: 快速但能看清
                #   - 0.05秒: 适中速度（当前设置）
                #   - 0.1秒:  较慢，每步都很清晰
                #   - 0.2秒:  很慢，适合演示讲解
                plt.pause(0.05)
        plt.pause(0.1)  # 所有节点绘制完成后的暂停时间

    def plot_path(self, path, cl='r', flag=False):
        """
        绘制路径（逐段动画显示）
        :param path: 路径节点列表
        :param cl: 路径颜色（默认红色）
        :param flag: 是否使用自定义颜色（False=使用红色，True=使用cl参数指定的颜色）
        """
        # 提取路径的 x 和 y 坐标
        path_x = [path[i][0] for i in range(len(path))]
        path_y = [path[i][1] for i in range(len(path))]

        # 选择颜色
        color = cl if flag else 'r'

        # ========== 路径动画显示控制 ==========
        # 逐段绘制路径，让路径绘制过程可见
        for i in range(len(path) - 1):
            # 绘制当前段（从第i个点到第i+1个点）
            plt.plot([path_x[i], path_x[i+1]], 
                     [path_y[i], path_y[i+1]], 
                     linewidth='3', color=color)
            
            # 暂停时间调整建议：
            #   - 0.01秒: 快速绘制
            #   - 0.05秒: 适中速度（当前设置）
            #   - 0.1秒:  较慢，能清晰看到每一段
            #   - 0.2秒:  很慢，适合演示
            plt.pause(0.05)

        # 重新绘制起点和终点（确保它们在路径上方显示）
        plt.plot(self.xI[0], self.xI[1], "bs")
        plt.plot(self.xG[0], self.xG[1], "gs")

        plt.pause(0.1)  # 路径绘制完成后的暂停时间

    def plot_visited_bi(self, v_fore, v_back):
        """
        绘制双向搜索的访问节点（动画显示）
        前向搜索和后向搜索同时进行，用不同颜色区分
        :param v_fore: 前向搜索（从起点开始）访问的节点列表
        :param v_back: 后向搜索（从终点开始）访问的节点列表
        """
        # 移除起点和终点（避免重复显示）
        if self.xI in v_fore:
            v_fore.remove(self.xI)

        if self.xG in v_back:
            v_back.remove(self.xG)

        len_fore, len_back = len(v_fore), len(v_back)

        # 同时绘制前向和后向搜索的节点
        for k in range(max(len_fore, len_back)):
            # 绘制前向搜索节点（灰色）
            if k < len_fore:
                plt.plot(v_fore[k][0], v_fore[k][1], linewidth='3', color='gray', marker='o')
            # 绘制后向搜索节点（蓝色）
            if k < len_back:
                plt.plot(v_back[k][0], v_back[k][1], linewidth='3', color='cornflowerblue', marker='o')

            # 按 ESC 键退出程序
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])

            # 每 10 个节点刷新一次显示
            if k % 10 == 0:
                plt.pause(0.001)
        plt.pause(0.01)

    @staticmethod
    def color_list():
        """
        颜色列表1：用于 ARA* 等多次迭代优化的算法
        返回两组颜色列表，分别用于访问节点和路径
        :return: (访问节点颜色列表, 路径颜色列表)
        """
        # 访问节点颜色列表（从浅到深）
        cl_v = ['silver',       # 银色
                'wheat',        # 小麦色
                'lightskyblue', # 浅天蓝
                'royalblue',    # 皇家蓝
                'slategray']    # 石板灰
        # 路径颜色列表（从浅到深）
        cl_p = ['gray',         # 灰色
                'orange',       # 橙色
                'deepskyblue',  # 深天蓝
                'red',          # 红色
                'm']            # 品红色
        return cl_v, cl_p

    @staticmethod
    def color_list_2():
        """
        颜色列表2：用于 LRTA* 等多次重规划的算法
        提供更多颜色选项，用于区分不同的搜索迭代
        :return: 颜色列表
        """
        cl = ['silver',           # 银色
              'steelblue',        # 钢青色
              'dimgray',          # 暗灰色
              'cornflowerblue',   # 矢车菊蓝
              'dodgerblue',       # 道奇蓝
              'royalblue',        # 皇家蓝
              'plum',             # 李子色
              'mediumslateblue',  # 中石板蓝
              'mediumpurple',     # 中紫色
              'blueviolet',       # 蓝紫色
              ]
        return cl
