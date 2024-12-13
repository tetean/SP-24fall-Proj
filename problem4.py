'''
@Author: tetean
@Time: 2024/12/13 5:50 PM
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
p_list = [0.1, 0.3, 0.5]  # 不同的 p 值
q_list = [0.1, 0.3, 0.5]  # 不同的 q 值
w_list = [0.1, 0.5, 1.0]  # 不同的 w 值
alpha_list = [0.5, 0.9, 0.99]  # 不同的折扣因子
max_t = 20
C = 1

def experiment(p, q, w, alpha):
    states = [(t, s) for t in range(max_t) for s in [0, 1]]
    actions = [0, 1]  # 动作：0 表示不访问，1 表示访问
    epsilon = 1e-6  # 收敛阈值

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

    def value_iteration():
        g = 0  # 初始长期平均开销
        h = {state: 0 for state in states}  # 偏差函数初始化
        policy = {state: 0 for state in states}  # 初始策略

        delta_list = []  # 用于记录每次迭代的最大变化量

        for iteration in range(1000):
            delta = 0
            new_h = h.copy()

            for state in states:
                t, s = state
                Q_values = []

                for u in actions:
                    Q = cost(t, s, u) - g
                    for t_next, s_next in states:
                        prob = transition_prob(t, s, u, t_next, s_next)
                        Q += alpha * prob * h[(t_next, s_next)]
                    Q_values.append(Q)

                new_h[state] = min(Q_values)
                policy[state] = actions[np.argmin(Q_values)]

                delta = max(delta, abs(new_h[state] - h[state]))

            h = new_h
            g = min(Q_values)  # 更新长期平均开销

            delta_list.append(delta)  # 记录当前迭代的最大变化量

            if delta < epsilon:
                break

        return g, h, policy

    # 运行值迭代算法
    g, h, optimal_policy = value_iteration()

    # 转换策略矩阵用于可视化
    optimal_policy_matrix = np.zeros((max_t, 2))  # 策略矩阵 (t, s)
    for state, action in optimal_policy.items():
        t, s = state
        optimal_policy_matrix[t, s] = action

    return optimal_policy_matrix[:10, :]

# 实验并可视化结果
fig, axes = plt.subplots(len(w_list), len(q_list), figsize=(15, 10), sharex=True, sharey=True)

for i, w in enumerate(w_list):
    for j, q in enumerate(q_list):
        policy_matrix = experiment(p=0.1, q=q, w=w, alpha=0.9)
        ax = axes[i, j]
        sns.heatmap(policy_matrix, annot=False, cmap="coolwarm", cbar=False, ax=ax)
        ax.set_title(f"w={w}, q={q}")
        ax.set_xlabel(r"信息源状态 ($ \tilde{s}_k $) ")
        ax.set_ylabel("时隙 ($t_k$)")


import matplotlib.patches as mpatches

# 添加图例
red_patch = mpatches.Patch(color="red", label="1 (访问)")
blue_patch = mpatches.Patch(color="blue", label="0 (不访问)")
fig.legend(handles=[blue_patch, red_patch], loc="upper center", ncol=2, fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局以避免图例与图形重叠

# plt.tight_layout()
plt.savefig("optimal_policy_w_q_experiment.png")
plt.show()
