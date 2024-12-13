'''
@Author: tetean
@Time: 2024/12/13 12:54 PM
@Info:
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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

# 参数设置
p = 0.1
q = 0.1
w = 0.1
C = 1
max_t = 20
states = [(t, s) for t in range(max_t) for s in [0, 1]]  # 假设最大时间间隔为 10
actions = [0, 1]  # 动作：0 表示不访问，1 表示访问
epsilon = 1e-6  # 收敛阈值
max_iterations = 1000  # 最大迭代次数

# 定义状态转移概率函数
def transition_prob(t, s, u, t_next, s_next):
    if u == 1:
        if t_next == 1 and s_next in [0, 1]:
            return p if s_next == 1 else 1 - p
        return 0
    else:
        if s == 0:
            if t_next == t + 1 and s_next == 0:
                return 1 - p
            elif t_next == 1 and s_next == 1:
                return p
        elif s == 1:
            if t_next == t + 1 and s_next == 1:
                return 1 - q
            elif t_next == 1 and s_next == 0:
                return q
        return 0

def cost(t, s, action):
    if action == 1:  # 访问信息源
        return C
    else:  # 不访问信息源
        if s == 0:
            return w * (t * (1 - p) ** t + sum((1 - p) ** (n - 1) * p * n for n in range(1, t + 1)))
        else:
            return w * (q * t + (1 - q) * t)

# 无穷时长平均开销最小化的值迭代算法
def average_cost_value_iteration(states, actions, transition_prob, cost, epsilon=1e-6, max_iterations=1000):
    g = 0  # 初始长期平均开销
    h = {state: 0 for state in states}  # 偏差函数初始化
    policy = {state: 0 for state in states}  # 初始策略

    delta_list = []  # 用于记录每次迭代的最大变化量

    for iteration in range(max_iterations):
        delta = 0
        new_h = h.copy()

        for state in states:
            t, s = state
            Q_values = []

            for u in actions:
                Q = cost(t, s, u) - g
                for t_next, s_next in states:
                    prob = transition_prob(t, s, u, t_next, s_next)
                    Q += prob * h[(t_next, s_next)]
                Q_values.append(Q)

            new_h[state] = min(Q_values)
            policy[state] = actions[np.argmin(Q_values)]

            delta = max(delta, abs(new_h[state] - h[state]))

        h = new_h
        g = min(Q_values)  # 更新长期平均开销

        delta_list.append(delta)  # 记录当前迭代的最大变化量
        print(f"Iteration {iteration}: g = {g}, delta = {delta}")

        if delta < epsilon:
            break

    return g, h, policy, delta_list

# 运行平均开销最小化的值迭代算法
g, h, optimal_policy, delta_list = average_cost_value_iteration(states, actions, transition_prob, cost, epsilon, max_iterations)

# 输出结果
print("\nOptimal Average Cost:")
print(f"g = {g}")

print("\nOptimal Bias Function (h):")
for state, value in h.items():
    print(f"State {state}: h = {value}")

print("\nOptimal Policy:")
for state, action in optimal_policy.items():
    print(f"State {state}: Action {action}")

# 使用更生动的表现形式可视化值函数收敛过程
plt.figure(figsize=(12, 6))
sns.lineplot(delta_list, marker='o', linestyle='-', color='b')
plt.xlabel("迭代次数", fontsize=14)
plt.ylabel("偏差函数$h$最大变化量", fontsize=14)
plt.title("值迭代收敛过程", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("value_iteration_convergence.png")
plt.show()

# 使用热图可视化最优策略
optimal_policy_matrix = np.zeros((10, 2))  # 策略矩阵 (t, s)
for state, action in optimal_policy.items():
    t, s = state
    if t >= 10: break
    optimal_policy_matrix[t, s] = action
# optimal_policy_matrix = optimal_policy_matrix[: 10,:]

# import matplotlib.patches as mpatches
#
# # 自定义图例
# action_0_patch = mpatches.Patch(color='blue', label="动作 0: 不访问")
# action_1_patch = mpatches.Patch(color='red', label="动作 1: 访问")
# plt.legend(handles=[action_0_patch, action_1_patch], loc='upper right', fontsize=12)

plt.figure(figsize=(10, 8))
sns.heatmap(optimal_policy_matrix, annot=True, cmap="coolwarm", cbar_kws={'label': "动作 (0: 不访问, 1: 访问)"}, linewidths=.5, linecolor='gray', cbar=False)
# plt.xticks([0.5, 1.5], ["状态 0", "状态 1"], fontsize=12)
plt.yticks(np.arange(0.5, 10.5), [f"t={t}" for t in range(10)], fontsize=12, rotation=0)
plt.title("最优策略矩阵", fontsize=16)
plt.xlabel(r"信息源状态 ($ \tilde{s}_k $) ", fontsize=14)
plt.ylabel("时隙 ($t_k$)", fontsize=14)
plt.tight_layout()
plt.savefig("optimal_policy_matrix_heatmap.png")
plt.show()
