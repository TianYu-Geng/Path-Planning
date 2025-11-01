"""
深度优先搜索算法 - 2D路径规划
纯DFS实现：使用栈结构，不考虑路径代价
@author: Ethan.Geng
"""

import os
import sys
import math

# 将搜索模块路径添加到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

from Search_2D import plotting, env
from Search_2D.Astar import AStar

class DFS(AStar):
    """
    DFS 深度优先搜索类（纯DFS实现）
    
    特点：
    - 使用栈结构（LIFO - 后进先出）
    - 不考虑路径代价，只关心是否访问过
    - 沿着一个方向一直走到底，遇到死路才回溯
    - 只允许上下左右4个方向移动
    - 找到的路径不一定是最短路径
    """
    def __init__(self, s_start, s_goal, heuristic_type):
        """
        初始化 DFS
        :param s_start: 起点
        :param s_goal: 终点
        :param heuristic_type: 启发式类型（DFS不使用，但保留接口兼容性）
        """
        super().__init__(s_start, s_goal, heuristic_type)
        # 重写移动方向：只允许上下左右4个方向
        # (x, y): 上(0,1), 下(0,-1), 左(-1,0), 右(1,0)
        self.u_set = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    
    def cost(self, s_start, s_goal):
        """
        重写代价函数：DFS 中所有移动的代价都为 1
        :param s_start: 起始节点
        :param s_goal: 目标节点
        :return: 移动代价（恒为1）
        """
        # 检查碰撞
        if self.is_collision(s_start, s_goal):
            return math.inf
        
        # DFS 核心：所有移动代价都是 1（只允许上下左右移动）
        return 1
    
    def searching(self, debug_steps=0):
        """
        深度优先搜索算法主函数（纯DFS版本）
        不考虑路径代价，只沿着一个方向走到底
        :param debug_steps: 打印前N步的调试信息，0表示不打印
        :return: path (路径列表), visited (访问顺序列表)
        """

        # 初始化
        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0  # 用于显示距离标注
        
        # 栈结构（使用列表模拟）
        stack = [self.s_start]
        
        # 已访问节点集合（纯DFS的核心）
        visited = set()
        visited.add(self.s_start)
        
        step = 0  # 步数计数

        if debug_steps > 0:
            print("=" * 80)
            print(f"DFS 前 {debug_steps} 步的详细执行过程")
            print("=" * 80)
            print(f"起点: {self.s_start}, 终点: {self.s_goal}")
            print(f"移动方向顺序: {self.u_set}")
            print(f"说明: 上(0,1), 下(0,-1), 左(-1,0), 右(1,0)")
            print("=" * 80 + "\n")

        # 主循环：栈不为空时继续搜索
        while stack:
            step += 1
            
            # 调试输出
            if debug_steps > 0 and step <= debug_steps:
                print(f"【步骤 {step}】")
                print(f"  栈内容（栈底→栈顶）: {stack}")
                print(f"  栈大小: {len(stack)}")
            
            # 从栈顶弹出节点（LIFO - 后进先出）
            s = stack.pop()
            
            if debug_steps > 0 and step <= debug_steps:
                print(f"  ★ POP出栈: {s}")
            
            # 记录访问顺序
            self.CLOSED.append(s)

            # 如果到达目标点，停止搜索
            if s == self.s_goal:
                if debug_steps > 0 and step <= debug_steps:
                    print(f"  🎯 找到目标！")
                break

            # 遍历当前节点的所有邻居节点
            pushed = []  # 记录本次压入栈的节点
            for s_n in self.get_neighbor(s):
                # 纯DFS逻辑：只检查是否访问过，不考虑代价
                if s_n not in visited and not self.is_collision(s, s_n):
                    # 标记为已访问（关键：在加入栈时就标记，避免重复加入）
                    visited.add(s_n)
                    
                    # 记录父节点（用于回溯路径）
                    self.PARENT[s_n] = s
                    
                    # 记录距离（仅用于可视化显示数字）
                    self.g[s_n] = self.g[s] + 1
                    
                    # 压入栈顶（LIFO）
                    stack.append(s_n)
                    pushed.append(s_n)
            
            if debug_steps > 0 and step <= debug_steps:
                if pushed:
                    print(f"  ↑ PUSH入栈: {pushed}")
                else:
                    print(f"  ⚠ 无可用邻居 → 回溯")
                print()  # 空行分隔
                
            if step == debug_steps:
                print("=" * 80)
                print(f"已显示前 {debug_steps} 步，继续搜索中...")
                print("=" * 80 + "\n")

        # 返回路径和访问顺序
        return self.extract_path(self.PARENT), self.CLOSED


def main():
    """
    主函数：演示 DFS 算法的使用
    """
    # 定义起点坐标
    s_start = (5, 5)
    # 定义终点坐标
    s_goal = (45, 25)

    # 创建 DFS 搜索对象
    # 参数：起点、终点、启发式函数类型（'None' 表示不使用启发式）
    dfs = DFS(s_start, s_goal, 'None')
    # 创建绘图对象，用于可视化搜索过程
    plot = plotting.Plotting(s_start, s_goal)

    # 执行 DFS 搜索，获取路径和访问顺序
    # debug_steps=20 表示打印前20步的详细信息
    # 设置为 0 则不打印任何调试信息
    path, visited = dfs.searching(debug_steps=20)
    
    # 去除重复访问的节点（DFS可能多次访问同一节点）
    visited = list(dict.fromkeys(visited))
    # 动画展示搜索过程和最终路径
    # 传递 dfs.g 代价字典，用于显示每个节点的距离标注
    plot.animation(path, visited, "Depth-first Searching (DFS)", cost_dict=dfs.g)


if __name__ == '__main__':
    main()
