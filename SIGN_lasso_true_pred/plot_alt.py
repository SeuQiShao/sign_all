import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from sklearn.metrics import r2_score
from scipy.stats import linregress
from scipy.stats import binned_statistic_2d
from scipy.optimize import curve_fit

Dynamics = 'enso'
dims = 0
true_path = 'true_{}_dim_{}.csv'.format(Dynamics,dims)
pred_path = 'pred_{}_dim_{}.csv'.format(Dynamics,dims)

# tensor(31.0537) tensor(20.9747) tensor(26.7891) tensor(1.3795)
# 读取数据
true_data0 = pd.read_csv(true_path, header=None)
pred_data0 = pd.read_csv(pred_path, header=None)
xy = pd.read_csv('enso_xy.csv', header=None)
print('data shape:', true_data0.shape)
# # max-min 逆归一化
# pred_data = pred_data0 * (30.8611 - 21.0302) + 21.0302
# true_data = true_data0 * (30.8611 - 21.0302) + 21.0302
# normalize 逆归一化
pred_data = pred_data0 * 1.3180 + 26.9876
true_data = true_data0 * 1.3180 + 26.9876

sep = 95

#mape
mape = np.mean(np.abs((true_data - pred_data) / true_data)) 
print('mape:', mape)
#train mape
train_mape = np.mean(np.abs((true_data.iloc[:sep,:] - pred_data.iloc[:sep,:]) / true_data.iloc[:sep,:])) 
print('train mape:', train_mape, 'max mape:', np.max(np.mean(np.abs((true_data.iloc[:sep,:] - pred_data.iloc[:sep,:]) / true_data.iloc[:sep,:]), 0)))
#test mape
test_mape = np.mean(np.abs((true_data.iloc[sep:,:] - pred_data.iloc[sep:,:]) / true_data.iloc[sep:,:])) 
print('test mape:', test_mape, 'max mape:', np.max(np.mean(np.abs((true_data.iloc[sep:,:] - pred_data.iloc[sep:,:]) / true_data.iloc[sep:,:]), 0)))

# 确保数据维度匹配
assert true_data.shape == pred_data.shape, "真实值和预测值的形状不匹配"

# 随机选取 5 条轨迹索引
np.random.seed(6434)  # 固定随机种子，保证可复现
#selected_indices = np.random.choice(true_data.shape[1], 1, replace=False)
selected_indices = [3]
# 颜色映射
colors = plt.cm.get_cmap('tab10', len(selected_indices))

# 绘制轨迹对比图
plt.figure(figsize=(8, 6))
for i, idx in enumerate(selected_indices):
    plt.plot(true_data.index, true_data.iloc[:, idx], linestyle='-', color=colors(i), label=f'True {idx}')
    plt.plot(pred_data.index, pred_data.iloc[:, idx], linestyle=':', color=colors(i), label=f'Pred {idx}')
plt.axvline(x=true_data.index[sep], color='red', linestyle='--', linewidth=1.5)
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('True vs Predicted Trajectories (5 Samples)')
plt.savefig('trajectory_comparison.png', dpi=300)


# if xy is not None:

#     # 计算每条轨迹的 MAPE
#     epsilon = 1e-8  # 防止除零
#     mape_per_traj = np.mean(np.abs((true_data - pred_data) / (true_data + epsilon)), axis=0)

#     fig, ax = plt.subplots(figsize=(15, 4))

#     # 设置 colormap
#     norm = plt.Normalize(np.nanmin(mape_per_traj), np.nanmax(mape_per_traj))
#     cmap = plt.cm.viridis

#     # 网格大小（5°间隔）
#     dx, dy = 0.25/3, 0.25/3  # 可根据数据密度调整小方块大小

#     # 在每个坐标位置画上小方块
#     for i in range(len(mape_per_traj)):
#         y = xy.iloc[i, 0]  # 经度（已经是转换后的）
#         x = 180 - xy.iloc[i, 1]  # 转换纬度
#         color = cmap(norm(mape_per_traj[i]))
#         rect = patches.Rectangle((x - dx/2, y - dy/2), dx, dy, color=color)
#         ax.add_patch(rect)

#     # 设置坐标轴
#     ax.set_xlim(180 - xy.iloc[:, 1].min(), 180 - xy.iloc[:, 1].max())  # 对应经度 120°E（170） 到 180°E（180）
#     ax.set_ylim(xy.iloc[:, 0].min(), xy.iloc[:, 0].max())  # 对应纬度 5°N（175） 到 5°S（185）
#     ax.set_xlabel('Longitude (°)')
#     ax.set_ylabel('Latitude (°)')
#     ax.set_title('Heatmap of MAPE over Trajectory Coordinates')

#     # 添加 colorbar
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])
#     cbar = plt.colorbar(sm, ax=ax)
#     cbar.set_label('MAPE')

#     plt.tight_layout()
#     plt.savefig('mape_heatmap.png', dpi=300)

# if xy is not None:

#     # 计算每条轨迹的 MAPE
#     epsilon = 1e-8  # 防止除零
#     mape_per_traj = np.mean(np.abs((true_data.iloc[:sep,:] - pred_data.iloc[:sep,:]) / (true_data.iloc[:sep,:] + epsilon)), axis=0)

#     fig, ax = plt.subplots(figsize=(15, 4))

#     # 设置 colormap
#     norm = plt.Normalize(np.nanmin(mape_per_traj), np.nanmax(mape_per_traj))
#     cmap = plt.cm.viridis

#     # 网格大小（5°间隔）
#     dx, dy = 0.25/3, 0.25/3  # 可根据数据密度调整小方块大小

#     # 在每个坐标位置画上小方块
#     for i in range(len(mape_per_traj)):
#         y = xy.iloc[i, 0]  # 经度（已经是转换后的）
#         x = 180 - xy.iloc[i, 1]  # 转换纬度
#         color = cmap(norm(mape_per_traj[i]))
#         rect = patches.Rectangle((x - dx/2, y - dy/2), dx, dy, color=color)
#         ax.add_patch(rect)

#     # 设置坐标轴
#     ax.set_xlim(180 - xy.iloc[:, 1].min(), 180 - xy.iloc[:, 1].max())  # 对应经度 120°E（170） 到 180°E（180）
#     ax.set_ylim(xy.iloc[:, 0].min(), xy.iloc[:, 0].max())  # 对应纬度 5°N（175） 到 5°S（185）
#     ax.set_xlabel('Longitude (°)')
#     ax.set_ylabel('Latitude (°)')
#     ax.set_title('Heatmap of MAPE over Trajectory Coordinates')

#     # 添加 colorbar
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])
#     cbar = plt.colorbar(sm, ax=ax)
#     cbar.set_label('MAPE')

#     plt.tight_layout()
#     plt.savefig('train_mape_heatmap.png', dpi=300)


# if xy is not None:

#     # 计算每条轨迹的 MAPE
#     epsilon = 1e-8  # 防止除零
#     mape_per_traj = np.mean(np.abs((true_data.iloc[sep:,:] - pred_data.iloc[sep:,:]) / (true_data.iloc[sep:,:] + epsilon)), axis=0)

#     fig, ax = plt.subplots(figsize=(15, 4))

#     # 设置 colormap
#     norm = plt.Normalize(np.nanmin(mape_per_traj), np.nanmax(mape_per_traj))
#     cmap = plt.cm.viridis

#     # 网格大小（5°间隔）
#     dx, dy = 0.25/3, 0.25/3  # 可根据数据密度调整小方块大小

#     # 在每个坐标位置画上小方块
#     for i in range(len(mape_per_traj)):
#         y = xy.iloc[i, 0]  # 经度（已经是转换后的）
#         x = 180 - xy.iloc[i, 1]  # 转换纬度
#         color = cmap(norm(mape_per_traj[i]))
#         rect = patches.Rectangle((x - dx/2, y - dy/2), dx, dy, color=color)
#         ax.add_patch(rect)

#     # 设置坐标轴
#     ax.set_xlim(180 - xy.iloc[:, 1].min(), 180 - xy.iloc[:, 1].max())  # 对应经度 120°E（170） 到 180°E（180）
#     ax.set_ylim(xy.iloc[:, 0].min(), xy.iloc[:, 0].max())  # 对应纬度 5°N（175） 到 5°S（185）
#     ax.set_xlabel('Longitude (°)')
#     ax.set_ylabel('Latitude (°)')
#     ax.set_title('Heatmap of MAPE over Trajectory Coordinates')

#     # 添加 colorbar
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])
#     cbar = plt.colorbar(sm, ax=ax)
#     cbar.set_label('MAPE')

#     plt.tight_layout()
#     plt.savefig('test_mape_heatmap.png', dpi=300)

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


# # ======== 读取数据 train========

# sep = 95
# true_pred = true_data.values.T[:,:sep]
# pred_pred = pred_data.values.T[:,:sep]
# print('true_pred shape:', true_pred.shape)



# # ========= C1：2D 直方图（代替散点图） =========
# true_flat = true_pred.flatten()
# pred_flat = pred_pred.flatten()


# # # 显示前先清洗 NaN 和极端异常值（如 <20 或 >32）
# # mask = (true_flat.flatten() > 20) & (true_flat.flatten() < 32)
# # true_flat = true_flat[mask]
# # pred_flat = pred_flat[mask]


# # 计算 R²
# ss_res = np.sum((true_flat - pred_flat)**2)
# ss_tot = np.sum((true_flat - np.mean(true_flat))**2)
# r2 = 1 - ss_res / ss_tot

# # 2D binning
# H, xedges, yedges = np.histogram2d(true_flat, pred_flat, bins=150)

# fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# ax = axs[0, 0]
# mesh = ax.pcolormesh(xedges, yedges, H.T, shading='auto', cmap='viridis', norm='linear')
# ax.plot([true_flat.min(), true_flat.max()], [true_flat.min(), true_flat.max()], 'r--')
# ax.set_xlim(true_flat.min(), true_flat.max())
# ax.set_ylim(pred_flat.min(), pred_flat.max())
# ax.set_xlabel('True SST')
# ax.set_ylabel('Predicted SST')
# ax.set_title(f'C1: True vs Predicted (R² = {r2:.3f})')
# fig.colorbar(mesh, ax=ax, label= 'Density')

# # ========= C2：每节点误差 vs 节点标准差 =========
# mape_node = np.mean(np.abs((pred_pred - true_pred) / true_pred), axis=1) * 100
# std_node = np.std(true_pred, axis=1)

# # 下采样或透明度散点图
# ax = axs[0, 1]
# perm = np.random.permutation(len(mape_node))[:10000]  # 取1万个点
# sc = ax.scatter(std_node[perm], mape_node[perm], s=2, alpha=0.3, c='tab:blue')
# ax.set_xlabel('SST Std per node')
# ax.set_ylabel('MAPE (%)')
# ax.set_title('C2: Node Error vs Variability')

# # 拿出下采样的横纵坐标
# x = std_node[perm]
# y = mape_node[perm]

# # 线性回归
# slope, intercept, r_value, p_value, std_err = linregress(x, y)

# # 生成拟合线
# x_fit = np.linspace(x.min(), x.max(), 100)
# y_fit = slope * x_fit + intercept

# # 添加到图上
# ax.plot(x_fit, y_fit, color='red', label=f'Linear Fit\n$y={slope:.2f}x + {intercept:.2f}$\n$R^2={r_value**2:.3f}$')
# ax.legend()


# # ========= C3：时间点误差随时间演化 =========
# mape_time = np.mean(np.abs((true_pred - pred_pred) / true_pred), axis=0) * 100
# print('Mean MAPE over time shape:', mape_time.shape)
# ax = axs[1, 0]
# ax.plot(range(1, len(mape_time)+1), mape_time, '-o', color='tab:orange')
# ax.axvline(x=sep, color='red', linestyle='--', linewidth=1.5)
# ax.set_xlabel('Forecast Month')
# ax.set_ylabel('Mean MAPE (%)')
# ax.set_title('C3: Forecast Error Over Time')
# ax.grid(True)

# # ========= C4：节点误差分布直方图 =========
# from scipy.stats import lognorm

# ax = axs[1, 1]
# # 画直方图（保持不变）
# ax.hist(mape_node, bins=20, color='tab:purple', edgecolor='black', alpha=0.8, density=True)

# # 添加均值和中位数参考线
# ax.axvline(np.mean(mape_node), color='r', linestyle='--', label=f'Mean = {np.mean(mape_node):.2f}%')
# ax.axvline(np.median(mape_node), color='g', linestyle='--', label=f'Median = {np.median(mape_node):.2f}%')

# # 设置轴标签和标题
# ax.set_xlabel('Node-wise MAPE (%)')
# ax.set_ylabel('Density')
# ax.set_title('C4: Histogram of Node Errors')

# # 拟合对数正态分布
# shape, loc, scale = lognorm.fit(mape_node, floc=0)  # 固定 loc=0，常用于误差拟合

# # 生成拟合曲线数据
# x_vals = np.linspace(min(mape_node), max(mape_node), 200)
# pdf_vals = lognorm.pdf(x_vals, shape, loc=loc, scale=scale)

# # 绘制对数高斯拟合曲线
# ax.plot(x_vals, pdf_vals, 'b-', label=f'Lognormal Fit\nσ={shape:.2f}, μ={np.log(scale):.2f}')

# ax.legend()

# plt.tight_layout()
# plt.savefig('train_SIGN_lasso_true_pred_analysis.png', dpi=300)
# plt.show()

# # # ======== 读取数据 test========

# sep = 95
# true_pred = true_data.values.T[:,sep:]
# pred_pred = pred_data.values.T[:,sep:]
# print('true_pred shape:', true_pred.shape)



# # ========= C1：2D 直方图（代替散点图） =========
# true_flat = true_pred.flatten()
# pred_flat = pred_pred.flatten()


# # 显示前先清洗 NaN 和极端异常值（如 <20 或 >32）
# mask = (true_flat.flatten() > 20) & (true_flat.flatten() < 32)
# true_flat = true_flat[mask]
# pred_flat = pred_flat[mask]


# 计算 R²
ss_res = np.sum((true_flat - pred_flat)**2)
ss_tot = np.sum((true_flat - np.mean(true_flat))**2)
r2 = 1 - ss_res / ss_tot

# 2D binning
H, xedges, yedges = np.histogram2d(true_flat, pred_flat, bins=150)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

ax = axs[0, 0]
mesh = ax.pcolormesh(xedges, yedges, H.T, shading='auto', cmap='viridis', norm='linear')
ax.plot([true_flat.min(), true_flat.max()], [true_flat.min(), true_flat.max()], 'r--')
ax.set_xlim(true_flat.min(), true_flat.max())
ax.set_ylim(pred_flat.min(), pred_flat.max())
ax.set_xlabel('True SST')
ax.set_ylabel('Predicted SST')
ax.set_title(f'C1: True vs Predicted (R² = {r2:.3f})')
fig.colorbar(mesh, ax=ax, label= 'Density')

# ========= C2：每节点误差 vs 节点标准差 =========
mape_node = np.mean(np.abs((pred_pred - true_pred) / true_pred), axis=1) * 100
std_node = np.std(true_pred, axis=1)

# 下采样或透明度散点图
ax = axs[0, 1]
perm = np.random.permutation(len(mape_node))[:10000]  # 取1万个点
sc = ax.scatter(std_node[perm], mape_node[perm], s=2, alpha=0.3, c='tab:blue')
ax.set_xlabel('SST Std per node')
ax.set_ylabel('MAPE (%)')
ax.set_title('C2: Node Error vs Variability')

# 拿出下采样的横纵坐标
x = std_node[perm]
y = mape_node[perm]

# 线性回归
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# 生成拟合线
x_fit = np.linspace(x.min(), x.max(), 100)
y_fit = slope * x_fit + intercept

# 添加到图上
ax.plot(x_fit, y_fit, color='red', label=f'Linear Fit\n$y={slope:.2f}x + {intercept:.2f}$\n$R^2={r_value**2:.3f}$')
ax.legend()


# ========= C3：时间点误差随时间演化 =========
mape_time = np.mean(np.abs((true_pred - pred_pred) / true_pred), axis=0) * 100
print('Mean MAPE over time shape:', mape_time.shape)
ax = axs[1, 0]
ax.plot(range(1, len(mape_time)+1), mape_time, '-o', color='tab:orange')
ax.axvline(x=sep, color='red', linestyle='--', linewidth=1.5)
ax.set_xlabel('Forecast Month')
ax.set_ylabel('Mean MAPE (%)')
ax.set_title('C3: Forecast Error Over Time')
ax.grid(True)

# ========= C4：节点误差分布直方图 =========
from scipy.stats import lognorm

ax = axs[1, 1]
# 画直方图（保持不变）
ax.hist(mape_node, bins=20, color='tab:purple', edgecolor='black', alpha=0.8, density=True)

# 添加均值和中位数参考线
ax.axvline(np.mean(mape_node), color='r', linestyle='--', label=f'Mean = {np.mean(mape_node):.2f}%')
ax.axvline(np.median(mape_node), color='g', linestyle='--', label=f'Median = {np.median(mape_node):.2f}%')

# 设置轴标签和标题
ax.set_xlabel('Node-wise MAPE (%)')
ax.set_ylabel('Density')
ax.set_title('C4: Histogram of Node Errors')

# 拟合对数正态分布
shape, loc, scale = lognorm.fit(mape_node, floc=0)  # 固定 loc=0，常用于误差拟合

# 生成拟合曲线数据
x_vals = np.linspace(min(mape_node), max(mape_node), 200)
pdf_vals = lognorm.pdf(x_vals, shape, loc=loc, scale=scale)

# 绘制对数高斯拟合曲线
ax.plot(x_vals, pdf_vals, 'b-', label=f'Lognormal Fit\nσ={shape:.2f}, μ={np.log(scale):.2f}')

ax.legend()

plt.tight_layout()
plt.savefig('test_SIGN_lasso_true_pred_analysis.png', dpi=300)
plt.show()


# # ======== 读取数据 ALL========


# sep = 96
# true_pred = true_data.values.T
# pred_pred = pred_data.values.T
# print('true_pred shape:', true_pred.shape)



# # ========= C1：2D 直方图（代替散点图） =========
# true_flat = true_pred.flatten()
# pred_flat = pred_pred.flatten()


# # # 显示前先清洗 NaN 和极端异常值（如 <20 或 >32）
# # mask = (true_flat.flatten() > 20) & (true_flat.flatten() < 32)
# # true_flat = true_flat[mask]
# # pred_flat = pred_flat[mask]


# # 计算 R²
# ss_res = np.sum((true_flat - pred_flat)**2)
# ss_tot = np.sum((true_flat - np.mean(true_flat))**2)
# r2 = 1 - ss_res / ss_tot

# # 2D binning
# H, xedges, yedges = np.histogram2d(true_flat, pred_flat, bins=150)

# fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# ax = axs[0, 0]
# mesh = ax.pcolormesh(xedges, yedges, H.T, shading='auto', cmap='viridis', norm='linear')
# ax.plot([true_flat.min(), true_flat.max()], [true_flat.min(), true_flat.max()], 'r--')
# ax.set_xlim(true_flat.min(), true_flat.max())
# ax.set_ylim(pred_flat.min(), pred_flat.max())
# ax.set_xlabel('True SST')
# ax.set_ylabel('Predicted SST')
# ax.set_title(f'C1: True vs Predicted (R² = {r2:.3f})')
# fig.colorbar(mesh, ax=ax, label= 'Density')

# # ========= C2：每节点误差 vs 节点标准差 =========
# mape_node = np.mean(np.abs((pred_pred - true_pred) / true_pred), axis=1) * 100
# std_node = np.std(true_pred, axis=1)

# # 下采样或透明度散点图
# ax = axs[0, 1]
# perm = np.random.permutation(len(mape_node))[:10000]  # 取1万个点
# sc = ax.scatter(std_node[perm], mape_node[perm], s=2, alpha=0.3, c='tab:blue')
# ax.set_xlabel('SST Std per node')
# ax.set_ylabel('MAPE (%)')
# ax.set_title('C2: Node Error vs Variability')

# # 拿出下采样的横纵坐标
# x = std_node[perm]
# y = mape_node[perm]

# # 线性回归
# slope, intercept, r_value, p_value, std_err = linregress(x, y)

# # 生成拟合线
# x_fit = np.linspace(x.min(), x.max(), 100)
# y_fit = slope * x_fit + intercept

# # 添加到图上
# ax.plot(x_fit, y_fit, color='red', label=f'Linear Fit\n$y={slope:.2f}x + {intercept:.2f}$\n$R^2={r_value**2:.3f}$')
# ax.legend()


# # ========= C3：时间点误差随时间演化 =========
# mape_time = np.mean(np.abs((true_pred - pred_pred) / true_pred), axis=0) * 100
# print('Mean MAPE over time shape:', mape_time.shape)
# ax = axs[1, 0]
# ax.plot(range(1, len(mape_time)+1), mape_time, '-o', color='tab:orange')
# ax.axvline(x=sep, color='red', linestyle='--', linewidth=1.5)
# ax.set_xlabel('Forecast Month')
# ax.set_ylabel('Mean MAPE (%)')
# ax.set_title('C3: Forecast Error Over Time')
# ax.grid(True)

# # ========= C4：节点误差分布直方图 =========
# from scipy.stats import lognorm

# ax = axs[1, 1]
# # 画直方图（保持不变）
# ax.hist(mape_node, bins=20, color='tab:purple', edgecolor='black', alpha=0.8, density=True)

# # 添加均值和中位数参考线
# ax.axvline(np.mean(mape_node), color='r', linestyle='--', label=f'Mean = {np.mean(mape_node):.2f}%')
# ax.axvline(np.median(mape_node), color='g', linestyle='--', label=f'Median = {np.median(mape_node):.2f}%')

# # 设置轴标签和标题
# ax.set_xlabel('Node-wise MAPE (%)')
# ax.set_ylabel('Density')
# ax.set_title('C4: Histogram of Node Errors')

# # 拟合对数正态分布
# shape, loc, scale = lognorm.fit(mape_node, floc=0)  # 固定 loc=0，常用于误差拟合

# # 生成拟合曲线数据
# x_vals = np.linspace(min(mape_node), max(mape_node), 200)
# pdf_vals = lognorm.pdf(x_vals, shape, loc=loc, scale=scale)

# # 绘制对数高斯拟合曲线
# ax.plot(x_vals, pdf_vals, 'b-', label=f'Lognormal Fit\nσ={shape:.2f}, μ={np.log(scale):.2f}')

# ax.legend()

# plt.tight_layout()
# plt.savefig('all_train_SIGN_lasso_true_pred_analysis.png', dpi=300)
# plt.show()

