'''
@ Author: tetean
@ Create time: 2024/12/11 16:59
@ Info:
'''

import numpy as np
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import font_manager

# 检查系统字体并自动选择可用的中文字体
available_fonts = set(f.name for f in font_manager.fontManager.ttflist)
chinese_fonts = ["PingFang SC", "SimHei", "Microsoft YaHei", "WenQuanYi Zen Hei"]
chosen_font = next((font for font in chinese_fonts if font in available_fonts), None)

if chosen_font is None:
    raise RuntimeError("未找到支持的中文字体，请安装如 PingFang SC 或 SimHei 的字体！")

# 设置字体为找到的中文字体
rcParams['font.sans-serif'] = [chosen_font]
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 参数设定
C = 1        # 访问时的固定能耗
w = 0.1      # 同步年龄权重
alpha = 0.999  # 折扣因子
p = 0.1      # 源状态从 0 -> 1 的转移概率
q = 0.1      # 源状态从 1 -> 0 的转移概率
max_t = 20   # 最大时间间隔
epsilon = 1e-6  # 收敛阈值

# 状态空间 (t, s) 其中 t 为时间间隔，s 为信息源状态
states = [(t, s) for t in range(1, max_t + 1) for s in [0, 1]]

# 初始化值函数 V 和最优策略 π
V = {state: 0 for state in states}  # 初始化值函数 V(x) 为 0
policy = {state: 0 for state in states}  # 初始化策略 π(x) 为 0

# 单步开销函数 g(x, u)
def cost(state, action):
    t, s = state
    if action == 1:  # 访问信息源
        return C
    else:  # 不访问信息源
        if s == 0:
            return w * (t * (1 - p) ** t + sum((1 - p) ** (n - 1) * p * n for n in range(1, t + 1)))
        else:
            return w * (q * t + (1 - q) * t)

# 转移概率 P(x' | x, u) 的实现
def transition_prob(state, action):
    t, s = state
    if action == 0:  # 不访问信息源
        next_state = (t + 1, s) if t + 1 <= max_t else (max_t, s)
        return {next_state: 1}
    else:  # 访问信息源
        if s == 0:
            return {(1, 0): 1 - p, (1, 1): p}
        else:
            return {(1, 0): q, (1, 1): 1 - q}

# 值迭代算法
def value_iteration():
    global V, policy
    iteration = 0
    delta_list = []  # 记录每次迭代的最大变化量
    while True:
        new_V = V.copy()
        delta = 0
        for state in states:
            action_costs = []
            for action in [0, 1]:  # 计算两种动作下的开销
                total_cost = cost(state, action)
                next_states_probs = transition_prob(state, action)
                next_cost = sum(prob * V[next_state] for next_state, prob in next_states_probs.items())
                action_costs.append(total_cost + alpha * next_cost)
            # 更新值函数和策略
            new_V[state] = min(action_costs)
            policy[state] = np.argmin(action_costs)  # 记录最优动作
            delta = max(delta, abs(new_V[state] - V[state]))
        V = new_V
        delta_list.append(delta)
        iteration += 1
        if delta < epsilon:  # 收敛条件
            break
    return iteration, delta_list

# 运行值迭代算法
iterations, delta_list = value_iteration()

# 可视化值函数收敛过程
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(delta_list) + 1), delta_list, marker='o')
plt.xlabel("迭代次数", fontsize=14)
plt.ylabel("值函数$V$最大变化量", fontsize=14)
plt.title("值迭代收敛过程", fontsize=16)
plt.grid(True)
plt.savefig("value_iteration_convergence.png")
plt.show()

# 可视化最优策略
optimal_policy = np.zeros((max_t, 2))  # 策略矩阵 (t, s)
for state, action in policy.items():
    t, s = state
    optimal_policy[t - 1, s] = action

plt.figure(figsize=(8, 6))
plt.imshow(optimal_policy, cmap="cool", origin="upper")
plt.colorbar(label="动作 (0: 不访问, 1: 访问)")
plt.xticks([0, 1], ["状态 0", "状态 1"], fontsize=12)
plt.yticks(range(max_t), [f"t={t}" for t in range(1, max_t + 1)], fontsize=12)
plt.title("最优策略矩阵", fontsize=16)
plt.xlabel("信息源状态", fontsize=14)
plt.ylabel("时间间隔 (t)", fontsize=14)
plt.savefig("optimal_policy_matrix.png")
plt.show()

# 可视化最优策略 (改进二值热图，增加块宽度)
plt.figure(figsize=(8, 6))

# 使用 imshow 绘制
plt.imshow(optimal_policy, cmap="coolwarm", interpolation="nearest", aspect="auto")

# 添加颜色栏和标签
plt.colorbar(label="动作 (0: 不访问, 1: 访问)")
plt.xticks([0, 1], ["状态 0", "状态 1"], fontsize=12)
plt.yticks(range(max_t), [f"t={t}" for t in range(1, max_t + 1)], fontsize=12)

# 设置标题和坐标轴标签
plt.title("最优策略矩阵", fontsize=16, pad=20)
plt.xlabel("信息源状态", fontsize=14)
plt.ylabel("时间间隔 (t)", fontsize=14)

# 保存并显示图像
plt.savefig("optimal_policy_matrix.png")
plt.show()

####################################
import matplotlib.patches as mpatches

# 可视化最优策略 (自定义图例)
plt.figure(figsize=(8, 6))

# 使用 imshow 绘制最优策略矩阵
plt.imshow(optimal_policy, cmap="coolwarm", interpolation="nearest", aspect="auto")

# 自定义图例
action_0_patch = mpatches.Patch(color='blue', label="动作 0: 不访问")
action_1_patch = mpatches.Patch(color='red', label="动作 1: 访问")
plt.legend(handles=[action_0_patch, action_1_patch], loc='upper right', fontsize=12)

# 添加坐标轴和标题
plt.xticks([0, 1], ["状态 0", "状态 1"], fontsize=12)
plt.yticks(range(max_t), [f"t={t}" for t in range(1, max_t + 1)], fontsize=12)
plt.title("最优策略矩阵", fontsize=16, pad=20)
plt.xlabel(r"信息源状态 ($ \tilde{s}_k $) ", fontsize=14)
plt.ylabel("时隙 ($t_k$)", fontsize=14)

# 保存并显示图像
plt.savefig("optimal_policy_matrix_custom_legend.png")
plt.show()



