import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

Dynamics = 'enso'
dims = 0
true_path = 'true_{}_dim_{}.csv'.format(Dynamics,dims)
pred_path = 'pred_{}_dim_{}.csv'.format(Dynamics,dims)
# tensor(31.0537) tensor(20.9747) tensor(26.7891) tensor(1.3795)
# 读取数据
true_data0 = pd.read_csv(true_path, header=None)
pred_data0 = pd.read_csv(pred_path, header=None)
xy = pd.read_csv('enso_xy.csv', header=None)
# # max-min 逆归一化
# pred_data = pred_data0 * (30.8611 - 21.0302) + 21.0302
# true_data = true_data0 * (30.8611 - 21.0302) + 21.0302
# normalize 逆归一化
pred_data = pred_data0 * 1.3785 + 26.7889
true_data = true_data0 * 1.3785 + 26.7889

#mape
mape = np.mean(np.abs((true_data - pred_data) / true_data)) * 100
print('mape:', mape)

# 确保数据维度匹配
assert true_data.shape == pred_data.shape, "真实值和预测值的形状不匹配"

# 随机选取 5 条轨迹索引
np.random.seed(94)  # 固定随机种子，保证可复现
selected_indices = np.random.choice(true_data.shape[1], 1, replace=False)

# 颜色映射
colors = plt.cm.get_cmap('tab10', len(selected_indices))

# 绘制轨迹对比图
plt.figure(figsize=(8, 6))
for i, idx in enumerate(selected_indices):
    plt.plot(true_data.index, true_data.iloc[:, idx], linestyle='-', color=colors(i), label=f'True {idx}')
    plt.plot(pred_data.index, pred_data.iloc[:, idx], linestyle=':', color=colors(i), label=f'Pred {idx}')

plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('True vs Predicted Trajectories (5 Samples)')
plt.savefig('trajectory_comparison.png', dpi=300)


if xy is not None:

    # 计算每条轨迹的 MAPE
    epsilon = 1e-8  # 防止除零
    mape_per_traj = np.mean(np.abs((true_data - pred_data) / (true_data + epsilon)), axis=0)

    fig, ax = plt.subplots(figsize=(15, 4))

    # 设置 colormap
    norm = plt.Normalize(np.nanmin(mape_per_traj), np.nanmax(mape_per_traj))
    cmap = plt.cm.viridis

    # 网格大小（5°间隔）
    dx, dy = 0.25/3, 0.25/3  # 可根据数据密度调整小方块大小

    # 在每个坐标位置画上小方块
    for i in range(len(mape_per_traj)):
        y = xy.iloc[i, 0]  # 经度（已经是转换后的）
        x = 180 - xy.iloc[i, 1]  # 转换纬度
        color = cmap(norm(mape_per_traj[i]))
        rect = patches.Rectangle((x - dx/2, y - dy/2), dx, dy, color=color)
        ax.add_patch(rect)

    # 设置坐标轴
    ax.set_xlim(180 - xy.iloc[:, 1].min(), 180 - xy.iloc[:, 1].max())  # 对应经度 120°E（170） 到 180°E（180）
    ax.set_ylim(xy.iloc[:, 0].min(), xy.iloc[:, 0].max())  # 对应纬度 5°N（175） 到 5°S（185）
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_title('Heatmap of MAPE over Trajectory Coordinates')

    # 添加 colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('MAPE')

    plt.tight_layout()
    plt.savefig('mape_heatmap.png', dpi=300)

# # 计算每条轨迹的 MSE
# mse_values = ((true_data - pred_data) ** 2).mean(axis=0)  # 计算每列（轨迹）的 MSE

# # 对 MSE 进行 log10 变换
# log_mse_values = np.log10(mse_values)

# # 绘制误差分布直方图
# plt.figure(figsize=(8, 6))
# plt.hist(log_mse_values, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
# plt.xlabel('Log10(MSE)')
# plt.ylabel('Frequency')
# plt.title('Log10(MSE) Distribution Across Trajectories')
# plt.grid(True, linestyle='--', alpha=0.6)

# plt.show()