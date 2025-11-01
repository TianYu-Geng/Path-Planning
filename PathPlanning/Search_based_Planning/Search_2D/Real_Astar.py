"""
A* 算法 - 2D路径规划（支持任意方向移动和曲线路径）
启发式搜索算法，结合实际代价和启发式函数

特点：
- 支持 8 方向移动（包括对角线）
- 代价 = 欧几里得距离 × 地形系数
- 支持三种启发式情况测试：h(n)=0, 过低估计, 过高估计
"""

import os
import sys
import math
import heapq
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

from Search_2D import plotting, env


class AStar:
    """
    A* 启发式搜索算法（带彩色地形代价，支持任意方向移动）
    
    核心特点：
    - 使用 f(n) = g(n) + w * h(n) 作为优先级
    - g(n): 从起点到当前节点的实际代价
    - h(n): 从当前节点到终点的启发式估计
    - w: 启发式权重（0.0=Dijkstra, 1.0=标准A*, >1.0=Weighted A*）
    - 保证找到最优路径（当 w * h(n) 可容许时）
    - 比 Dijkstra 更快（有启发式引导）
    
    移动方式：
    - 支持 8 方向移动（包括对角线）
    - 代价 = 欧几里得距离 × 地形系数
    - 可以形成平滑的曲线路径
    
    代价模型（彩色可视化）：
    - 灰色区域（默认）：代价系数 = 1
    - 🔴 红色区域：代价系数 = 2（40个节点）
    - 🟡 黄色区域：代价系数 = 3（35个节点）
    - 🔵 蓝色区域：代价系数 = 4（25个节点）
    - 🟢 绿色区域：代价系数 = 5（30个节点）
    - ⭐ 特殊：起点和终点周围距离3的一圈，随机高代价(2-5)
    
    起点/终点周围的随机高代价圈能打乱累计代价的连贯性，
    避免结果呈现过于规律的 1→2→3→4→5 递增模式。
    A* 会智能地结合启发式函数和实际代价，选择最优路径！
    """
    
    def __init__(self, s_start, s_goal, heuristic_type, heuristic_weight=1.0):
        """
        初始化 A* 算法
        :param s_start: 起点坐标
        :param s_goal: 终点坐标
        :param heuristic_type: 启发式类型（'manhattan' 或 'euclidean'）
        :param heuristic_weight: 启发式权重
            - 0.0: h(n) = 0，等价于 Dijkstra
            - 1.0: h(n)，标准 A*（可容许，保证最优）
            - >1.0: w*h(n)，过高估计（Weighted A*，更快但不保证最优）
        """
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type
        self.heuristic_weight = heuristic_weight  # 启发式权重

        self.Env = env.Env()  # class Env

        # A* 允许 8 方向移动（包括对角线）
        # 上、上右、右、右下、下、下左、左、左上
        self.u_set = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                      (1, 0), (1, -1), (0, -1), (-1, -1)]
        self.obs = self.Env.obs  # position of obstacles

        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come (实际代价)
        
        # 设置随机种子
        random.seed(42)
        
        # 预设特定节点的地形代价（用于可视化）
        # 代价值：1（灰色，默认），2（红色），3（黄色），4（蓝色），5（绿色）
        self.terrain_cost = {}
        self.terrain_colors = {}  # 节点颜色映射
        
        # 生成不同代价区域（每种代价不同数量，起点/终点周围一圈）
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

    def searching(self):
        """
        A* 搜索算法主函数
        
        A* 的核心：优先级 f(n) = g(n) + h(n)
        - g(n): 从起点到当前节点的实际代价（考虑地形）
        - h(n): 从当前节点到终点的启发式估计
        - 总是扩展 f(n) 最小的节点
        
        :return: path (路径列表), visited (访问顺序列表)
        """

        # 初始化起点
        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0  # 起点的实际代价为0
        self.g[self.s_goal] = math.inf  # 终点初始代价为无穷大
        
        # 将起点加入优先队列，优先级为 f(n) = g(n) + h(n)
        # 注意：这里使用 f(n)，而 Dijkstra 只用 g(n)
        heapq.heappush(self.OPEN,
                       (self.f_value(self.s_start), self.s_start))

        # 主循环：优先队列不为空时继续搜索
        while self.OPEN:
            # 弹出 f(n) 最小的节点（优先级 = g(n) + h(n)）
            _, s = heapq.heappop(self.OPEN)
            # 记录访问顺序
            self.CLOSED.append(s)

            # 如果到达目标点，停止搜索
            if s == self.s_goal:  # stop condition
                break

            # 遍历当前节点的所有邻居
            for s_n in self.get_neighbor(s):
                # ========== A* 实际代价计算 ==========
                # 计算从起点经过当前节点到邻居的实际代价
                # new_cost = g(s) + cost(s, s_n)
                # 这里 cost(s, s_n) 使用地形代价
                new_cost = self.g[s] + self.cost(s, s_n)

                # 如果邻居节点未访问过，初始化其代价
                if s_n not in self.g:
                    self.g[s_n] = math.inf

                # 如果找到更短的路径，更新节点信息
                if new_cost < self.g[s_n]:  # conditions for updating Cost
                    # 更新邻居节点的实际代价
                    self.g[s_n] = new_cost
                    # 记录父节点
                    self.PARENT[s_n] = s

                    # ========== A* 核心：优先级 = f(n) = g(n) + w * h(n) ==========
                    # 将节点加入优先队列，优先级为 f(n) = g(n) + w * h(n)
                    # 启发式权重 w 的影响：
                    # - w = 0.0:   priority = g(n)，等价于 Dijkstra
                    # - w = 1.0:   priority = g(n) + h(n)，标准 A*（保证最优）
                    # - w > 1.0:  priority = g(n) + w*h(n)，Weighted A*（更快，但不保证最优）
                    heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

        # 返回路径和访问顺序
        return self.extract_path(self.PARENT), self.CLOSED

    def searching_repeated_astar(self, e):
        """
        repeated A*.
        :param e: weight of A*
        :return: path and visited order
        """

        path, visited = [], []

        while e >= 1:
            p_k, v_k = self.repeated_searching(self.s_start, self.s_goal, e)
            path.append(p_k)
            visited.append(v_k)
            e -= 0.5

        return path, visited

    def repeated_searching(self, s_start, s_goal, e):
        """
        run A* with weight e.
        :param s_start: starting state
        :param s_goal: goal state
        :param e: weight of a*
        :return: path and visited order.
        """

        g = {s_start: 0, s_goal: float("inf")}
        PARENT = {s_start: s_start}
        OPEN = []
        CLOSED = []
        heapq.heappush(OPEN,
                       (g[s_start] + e * self.heuristic(s_start), s_start))

        while OPEN:
            _, s = heapq.heappop(OPEN)
            CLOSED.append(s)

            if s == s_goal:
                break

            for s_n in self.get_neighbor(s):
                new_cost = g[s] + self.cost(s, s_n)

                if s_n not in g:
                    g[s_n] = math.inf

                if new_cost < g[s_n]:  # conditions for updating Cost
                    g[s_n] = new_cost
                    PARENT[s_n] = s
                    heapq.heappush(OPEN, (g[s_n] + e * self.heuristic(s_n), s_n))

        return self.extract_path(PARENT), CLOSED

    def get_neighbor(self, s):
        """
        find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """

        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]

    def cost(self, s_start, s_goal):
        """
        代价函数：欧几里得距离 × 地形代价系数
        支持任意方向移动，包括对角线，形成更平滑的曲线路径
        
        :param s_start: 起始节点
        :param s_goal: 目标节点
        :return: 移动代价（欧几里得距离 × 地形系数）
        """
        # 检查碰撞
        if self.is_collision(s_start, s_goal):
            return math.inf
        
        # ========== A* 使用地形代价 × 距离 ==========
        # 1. 基础距离（欧几里得距离）
        #    - 直线移动（上下左右）: 1.0
        #    - 对角线移动: √2 ≈ 1.414
        #    这样可以形成更平滑的曲线路径
        base_distance = math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])
        
        # 2. 地形代价系数
        #    - 灰色区域：1（默认）
        #    - 红色区域：2
        #    - 黄色区域：3
        #    - 蓝色区域：4
        #    - 绿色区域：5
        terrain_factor = self.get_terrain_cost(s_goal)
        
        # 3. 最终代价 = 基础距离 × 地形系数
        #    这样支持任意方向移动，形成平滑曲线路径
        final_cost = base_distance * terrain_factor
        
        return final_cost

    def is_collision(self, s_start, s_end):
        """
        check if the line segment (s_start, s_end) is collision.
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """

        if s_start in self.obs or s_end in self.obs:
            return True

        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True

        return False

    def f_value(self, s):
        """
        计算 f(n) = g(n) + w * h(n)
        :param s: 当前节点
        :return: f值（优先级）
        """
        # f(n) = g(n) + w * h(n)
        # w = 0.0: h(n) = 0，等价于 Dijkstra
        # w = 1.0: 标准 A*（可容许，保证最优）
        # w > 1.0: Weighted A*（过高估计，更快但不保证最优）
        return self.g[s] + self.heuristic_weight * self.heuristic(s)

    def extract_path(self, PARENT):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.s_goal]
        s = self.s_goal

        while True:
            s = PARENT[s]
            path.append(s)

            if s == self.s_start:
                break

        return list(path)

    def heuristic(self, s):
        """
        Calculate heuristic.
        :param s: current node (state)
        :return: heuristic function value
        """

        heuristic_type = self.heuristic_type  # heuristic type
        goal = self.s_goal  # goal node

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])


def main():
    """
    主函数：测试三种启发式情况
    1. h(n) = 0 (weight=0.0): 等价于 Dijkstra
    2. 过低估计 (weight=1.0): 标准 A*，保证最优
    3. 过高估计 (weight=2.5): Weighted A*，更快但不保证最优
    """
    # 定义起点坐标
    s_start = (5, 5)
    # 定义终点坐标
    s_goal = (45, 25)
    
    # ========== 三种启发式情况测试 ==========
    test_cases = [
        {
            "name": "情况1: h(n) = 0",
            "weight": 0.0,
            "description": "等价于 Dijkstra，只考虑实际代价 g(n)"
        },
        {
            "name": "情况2: 过低估计（标准 A*）",
            "weight": 1.0,
            "description": "h(n) ≤ 真实代价，保证找到最优路径"
        },
        {
            "name": "情况3: 过高估计（Weighted A*）",
            "weight": 2.5,
            "description": "w * h(n) > 真实代价，更快但不保证最优"
        }
    ]
    
    print("=" * 100)
    print("A* 算法三种启发式情况对比测试")
    print("=" * 100)
    print(f"起点: {s_start}, 终点: {s_goal}\n")
    
    results = []
    
    # 测试每种情况
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*100}")
        print(f"【{case['name']}】")
        print(f"{'='*100}")
        print(f"说明: {case['description']}")
        print(f"启发式权重 w = {case['weight']}")
        print(f"优先级 f(n) = g(n) + {case['weight']} * h(n)")
        
        # 创建 A* 搜索对象
        astar = AStar(s_start, s_goal, "euclidean", heuristic_weight=case['weight'])
        plot = plotting.Plotting(s_start, s_goal)
        
        # 执行搜索
        path, visited = astar.searching()
        
        # 计算路径总代价（从起点到终点的总代价）
        # g[s_goal] 存储了从起点到终点的实际代价
        total_cost = astar.g[s_goal] if s_goal in astar.g and astar.g[s_goal] != math.inf else float('inf')
        
        # 保存结果
        results.append({
            "name": case['name'],
            "weight": case['weight'],
            "visited_count": len(visited),
            "path_length": len(path),
            "total_cost": total_cost
        })
        
        print(f"\n搜索结果:")
        print(f"  - 访问节点数: {len(visited)}")
        print(f"  - 路径长度: {len(path)}")
        print(f"  - 路径总代价: {total_cost:.2f}" if total_cost != float('inf') else "  - 路径总代价: 未找到路径")
        
        # 打印地形信息（只在第一次打印）
        if i == 1:
            print(f"\n地形代价分布：")
            print(f"  灰色区域（默认）: 代价 = 1")
            print(f"  红色区域: 代价 = 2, 数量 = {sum(1 for v in astar.terrain_cost.values() if v == 2)}")
            print(f"  黄色区域: 代价 = 3, 数量 = {sum(1 for v in astar.terrain_cost.values() if v == 3)}")
            print(f"  蓝色区域: 代价 = 4, 数量 = {sum(1 for v in astar.terrain_cost.values() if v == 4)}")
            print(f"  绿色区域: 代价 = 5, 数量 = {sum(1 for v in astar.terrain_cost.values() if v == 5)}")
            print(f"\n  ⭐ 特殊设置：")
            print(f"    - 起点 {s_start} 周围距离3的一圈：随机高代价(2-5)")
            print(f"    - 终点 {s_goal} 周围距离3的一圈：随机高代价(2-5)")
        
        # 动画展示（每个情况都显示）
        plot.animation(path, visited, f"A*: {case['name']} (w={case['weight']})", 
                       cost_dict=astar.g, 
                       terrain_colors=astar.terrain_colors)
    
    # ========== 重要说明 ==========
    print(f"\n{'='*100}")
    print("【重要说明】")
    print(f"{'='*100}")
    print("w (权重) 的含义：")
    print("  w = 启发式权重（heuristic_weight），控制启发式函数在优先级计算中的影响")
    print("  - w = 0.0: 不使用启发式，等价于 Dijkstra 算法")
    print("  - w = 1.0: 标准 A*，启发式函数按原始值使用")
    print("  - w > 1.0: Weighted A*，过度重视启发式，更快但可能不最优")
    print("\n优先级计算公式：")
    print("  f(n) = g(n) + w * h(n)")
    print("  - g(n): 从起点到当前节点的累计实际代价（考虑地形）")
    print("  - h(n): 从当前节点到终点的启发式估计（欧几里得距离）")
    print("  - w: 启发式权重")
    print("\n圆圈中数字的含义：")
    print("  显示的是 g(n) 值：从起点到该节点的累计实际代价")
    print("  - 起点: g = 0")
    print("  - 后续节点: g = 从起点到该节点的路径上所有移动代价之和")
    print("  - 例如：起点→节点A(代价2)→节点B(代价3)，则节点B的g = 2+3 = 5")
    print("=" * 100)
    
    # 对比总结
    print(f"\n{'='*100}")
    print("【对比总结】")
    print(f"{'='*100}")
    print(f"{'情况':<30} {'权重w':<10} {'访问节点':<12} {'路径长度':<12} {'总代价g(终点)':<12}")
    print("-" * 100)
    for r in results:
        cost_str = f"{r['total_cost']:.2f}" if r['total_cost'] != float('inf') else "未找到"
        print(f"{r['name']:<30} {r['weight']:<10.1f} {r['visited_count']:<12} {r['path_length']:<12} {cost_str:<12}")


if __name__ == '__main__':
    main()
