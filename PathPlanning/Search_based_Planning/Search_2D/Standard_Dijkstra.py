"""
Dijkstra 算法 - 2D路径规划
单源最短路径算法，保证找到最优解
@author: huiming zhou
"""

import os
import sys
import math
import heapq
import random

# 将搜索模块路径添加到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

from Search_2D import plotting, env
from Search_2D.Astar import AStar


class Dijkstra(AStar):
    """
    Dijkstra 最短路径算法（彩色地形代价）
    
    核心特点：
    - 使用实际代价 g(n) 作为优先级（不使用启发式函数）
    - 总是扩展代价最小的节点
    - 保证找到最优路径
    - 比 A* 慢，但不需要启发式函数
    
    代价模型（彩色可视化）：
    - 灰色区域（默认）：代价 = 1
    - 🔴 红色区域：代价 = 2（40个节点）
    - 🟡 黄色区域：代价 = 3（35个节点）
    - 🔵 蓝色区域：代价 = 4（25个节点）
    - 🟢 绿色区域：代价 = 5（30个节点）
    - ⭐ 特殊：起点和终点周围距离3的一圈，随机高代价(2-5)
    
    起点/终点周围的随机高代价圈能打乱累计代价的连贯性，
    避免结果呈现过于规律的 1→2→3→4→5 递增模式。
    Dijkstra 会智能地绕开高代价区域，选择总代价最小的路径！
    
    与其他算法的比较：
    - BFS: 不考虑代价，按层扩展
    - Dijkstra: 考虑代价，按代价从小到大扩展（每条边代价不同）
    - A*: 使用 f(n) = g(n) + h(n)，有启发式引导
    """
    
    def __init__(self, s_start, s_goal, heuristic_type):
        """
        初始化 Dijkstra
        :param s_start: 起点
        :param s_goal: 终点
        :param heuristic_type: 启发式类型（Dijkstra不使用，但保留接口兼容性）
        """
        super().__init__(s_start, s_goal, heuristic_type)
        # Dijkstra 只允许 4 方向移动（不允许对角线）
        # 上、下、左、右
        self.u_set = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        
        # 设置随机种子
        random.seed(42)
        
        # 预设特定节点的地形代价（用于可视化）
        # 代价值：1（灰色，默认），2（红色），3（黄色），4（蓝色），5（绿色）
        self.terrain_cost = {}
        self.terrain_colors = {}  # 节点颜色映射
        
        # 生成不同代价区域（每种代价15个节点）
        self._initialize_terrain()
    
    def _initialize_terrain(self):
        """
        初始化地形代价区域
        在地图上随机放置不同代价的节点
        """
        # 首先在起点和终点周围距离为3的一圈设置随机高代价
        self._set_surrounding_costs(self.s_start, distance=3)
        self._set_surrounding_costs(self.s_goal, distance=3)
        
        # 代价配置：(代价值, 颜色, 数量)
        terrain_types = [
            (2, 'red', 40),      # 代价2：红色，40个节点
            (3, 'yellow', 35),   # 代价3：黄色，35个节点
            (4, 'blue', 25),     # 代价4：蓝色，25个节点
            (5, 'green', 30)     # 代价5：绿色，30个节点
        ]
        
        # 地图范围（避开边界）
        x_range = range(8, 43)  # x: 8~42
        y_range = range(3, 28)  # y: 3~27
        
        # 为每种地形类型随机选择位置
        for cost_value, color, count in terrain_types:
            placed = 0
            attempts = 0
            while placed < count and attempts < 1000:
                x = random.choice(list(x_range))
                y = random.choice(list(y_range))
                node = (x, y)
                
                # 确保不重复，不在起点终点，不在障碍物
                if (node not in self.terrain_cost and 
                    node != self.s_start and 
                    node != self.s_goal and
                    node not in self.obs):
                    self.terrain_cost[node] = cost_value
                    self.terrain_colors[node] = color
                    placed += 1
                
                attempts += 1
    
    def _set_surrounding_costs(self, center, distance=3):
        """
        在指定中心点周围距离为distance的一圈设置随机高代价
        :param center: 中心点坐标
        :param distance: 距离（曼哈顿距离）
        """
        cx, cy = center
        # 获取距离为distance的所有节点（曼哈顿距离）
        for dx in range(-distance, distance + 1):
            for dy in range(-distance, distance + 1):
                # 曼哈顿距离 = |dx| + |dy|
                if abs(dx) + abs(dy) == distance:
                    node = (cx + dx, cy + dy)
                    # 确保不在障碍物上，不是起点或终点
                    if (node not in self.obs and 
                        node != self.s_start and 
                        node != self.s_goal):
                        # 随机选择高代价 (2-5)
                        cost = random.choice([2, 3, 4, 5])
                        self.terrain_cost[node] = cost
                        
                        # 根据代价设置颜色
                        color_map = {2: 'red', 3: 'yellow', 4: 'blue', 5: 'green'}
                        self.terrain_colors[node] = color_map[cost]
    
    def get_terrain_cost(self, node):
        """
        获取某个节点的地形代价
        :param node: 节点坐标
        :return: 地形代价（1, 2, 3, 4, 5）
        """
        # 如果是预设的特殊地形，返回预设代价
        if node in self.terrain_cost:
            return self.terrain_cost[node]
        # 否则返回默认代价 1（灰色区域）
        return 1
    
    def cost(self, s_start, s_goal):
        """
        代价函数：直接返回目标节点的地形代价
        这是 Dijkstra 与 BFS 的关键区别！
        
        :param s_start: 起始节点
        :param s_goal: 目标节点
        :return: 移动代价（1, 2, 3, 4, 5）
        """
        # 检查碰撞
        if self.is_collision(s_start, s_goal):
            return math.inf
        
        # ========== Dijkstra 的优势：不同位置有不同代价 ==========
        # 直接返回目标节点的地形代价
        # - 灰色区域：代价 = 1（默认）
        # - 红色区域：代价 = 2
        # - 黄色区域：代价 = 3
        # - 蓝色区域：代价 = 4
        # - 绿色区域：代价 = 5
        # 
        # Dijkstra 会自动绕开高代价区域！
        return self.get_terrain_cost(s_goal)
    
    def searching(self):
        """
        Dijkstra 搜索算法主函数
        :return: path (路径列表), visited (访问顺序列表)
        """

        # 初始化起点
        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0  # 起点的代价为0
        self.g[self.s_goal] = math.inf  # 终点初始代价为无穷大
        
        # 将起点加入优先队列，优先级为实际代价 g(n)
        # 注意：这里只用 g(n)，没有 h(n)，这是与 A* 的关键区别！
        heapq.heappush(self.OPEN, (0, self.s_start))

        # 主循环：优先队列不为空时继续搜索
        while self.OPEN:
            # 弹出代价最小的节点（优先级 = g(n)）
            _, s = heapq.heappop(self.OPEN)
            # 记录访问顺序
            self.CLOSED.append(s)

            # 如果到达目标点，停止搜索
            if s == self.s_goal:
                break

            # 遍历当前节点的所有邻居
            for s_n in self.get_neighbor(s):
                # ========== 代价函数计算 ==========
                # 计算从起点经过当前节点到邻居的实际代价
                # new_cost = g(s) + cost(s, s_n)
                # 这就是 Dijkstra 的核心：只考虑实际代价 g(n)
                new_cost = self.g[s] + self.cost(s, s_n)

                # 如果邻居节点未访问过，初始化其代价
                if s_n not in self.g:
                    self.g[s_n] = math.inf

                # 如果找到更短的路径，更新节点信息
                if new_cost < self.g[s_n]:
                    # 更新邻居节点的代价
                    self.g[s_n] = new_cost
                    # 记录父节点
                    self.PARENT[s_n] = s

                    # ========== Dijkstra 核心：优先级 = g(n) ==========
                    # 将节点加入优先队列，优先级为实际代价 new_cost
                    # 对比：
                    # - Dijkstra: priority = g(n)              <- 这里
                    # - A*:       priority = f(n) = g(n) + h(n)
                    # - BFS:      priority = 常数（FIFO队列）
                    heapq.heappush(self.OPEN, (new_cost, s_n))

        # 返回路径和访问顺序
        return self.extract_path(self.PARENT), self.CLOSED


def main():
    """
    主函数：演示 Dijkstra 算法的使用
    """
    # 定义起点坐标
    s_start = (5, 5)
    # 定义终点坐标
    s_goal = (45, 25)

    # 创建 Dijkstra 搜索对象
    # 参数：起点、终点、启发式函数类型（Dijkstra不使用启发式，传'None'）
    dijkstra = Dijkstra(s_start, s_goal, 'None')
    # 创建绘图对象，用于可视化搜索过程
    plot = plotting.Plotting(s_start, s_goal)

    # 执行 Dijkstra 搜索，获取路径和访问顺序
    path, visited = dijkstra.searching()
    
    # 打印地形信息
    print("=" * 80)
    print("地形代价分布：")
    print(f"  灰色区域（默认）: 代价 = 1")
    print(f"  红色区域: 代价 = 2, 数量 = {sum(1 for v in dijkstra.terrain_cost.values() if v == 2)}")
    print(f"  黄色区域: 代价 = 3, 数量 = {sum(1 for v in dijkstra.terrain_cost.values() if v == 3)}")
    print(f"  蓝色区域: 代价 = 4, 数量 = {sum(1 for v in dijkstra.terrain_cost.values() if v == 4)}")
    print(f"  绿色区域: 代价 = 5, 数量 = {sum(1 for v in dijkstra.terrain_cost.values() if v == 5)}")
    print(f"\n  ⭐ 特殊设置：")
    print(f"    - 起点 {s_start} 周围距离3的一圈：随机高代价(2-5)")
    print(f"    - 终点 {s_goal} 周围距离3的一圈：随机高代价(2-5)")
    print(f"    - 作用：打乱累计代价的连贯性")
    print("=" * 80)
    
    # 动画展示搜索过程和最终路径
    # 传递地形颜色和代价字典
    plot.animation(path, visited, "Dijkstra's Algorithm", 
                   cost_dict=dijkstra.g, 
                   terrain_colors=dijkstra.terrain_colors)


if __name__ == '__main__':
    main()
