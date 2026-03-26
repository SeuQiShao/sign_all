from tqdm import tqdm
from utils_file import arg_parser, data_loader
from model.modules import *
from model import utils, model_loader
import random
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuitCV, LassoCV, ARDRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.feature_selection import mutual_info_regression
from joblib import Parallel, delayed
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from collections import Counter
from sklearn.cluster import DBSCAN

def lasso_AIC(model, x, y):
    mse = np.mean((model.predict(x) - y)**2)
    if hasattr(model, 'intercept_'):
        p = (np.abs(model.coef_) > 0).sum() + np.sum(np.abs(model.intercept_) > 0)
    else:
        p = (np.abs(model.coef_) > 0).sum()
    aic = x.shape[0] * np.log(mse) + 2 * p
    #bic = p * np.log(x.shape[0]) - x.shape[0] * np.log(mse)
    return aic

def compute_node_features(data, i, edge_index, t, args, device):
    neighbor_i = edge_index[0][edge_index[1] == i]
    neighbor_num = len(neighbor_i)

    # if neighbor_num > args.lasso_neighbor_num:
    #     neighbor_index = random.sample(list(neighbor_i), args.lasso_neighbor_num)
    #     neighbor_coef = neighbor_num / args.lasso_neighbor_num
    # else:
    #     neighbor_index = neighbor_i
    #     neighbor_coef = 1

    neighbor_index = neighbor_i
    neighbor_coef = 1

    x_neighbor = data[neighbor_index, :, args.k]
    x_i = data[i, :, args.k]

    if neighbor_num == 0:
        # 如果没有邻居，直接返回空张量
        print(f"Node {i} has no neighbors, skipping.")
        x_c = None
    else:
        # 计算耦合动力学
        x_c = 0
        for j in x_neighbor:
            x_c += utils.coupled_fun_lib(x_i.reshape(-1, 1), j.reshape(-1, 1), 
                                         args.poly_p, args.poly_n, device, activate=args.activate)
        if args.agg == 'mean':
            x_c = x_c / x_neighbor.shape[0]
        x_c = x_c * neighbor_coef

    # 使用五点公式计算自身动力学导数
    x_vals = data[i, :, args.k].cpu().numpy()  # 转为 NumPy 方便操作
    t_vals = t.cpu().numpy()  # 时间点
    dt = t_vals[1] - t_vals[0]  # 假设均匀时间步长

    # 五点公式计算导数（减少前两个和后两个点）
    x_dot_vals = (-x_vals[4:] + 8 * x_vals[3:-1] - 8 * x_vals[1:-3] + x_vals[:-4]) / (12 * dt)
    x_dot = torch.tensor(x_dot_vals, device=device)

    # 更新 x1 和 x_c 的长度
    x1 = utils.fun_lib(data[i, :, args.k].reshape(-1, 1), args.poly_p, args.poly_n, device, activate=args.activate)
    x1 = x1[2:-2, :]  # 截取中间部分，与五点公式保持一致

    if x_c is not None:
        x_c = x_c[2:-2, :]  # 截取中间部分，与五点公式保持一致
        x_data = torch.cat((x1[:,1:], x_c), 1)
    else:
        x_data = x1[:,1:]
    #index = x_dot.abs()>1e-3
    return x_data, x_dot

def lasso_fit_single_node(x_data, x_dot, alpha=0.01):
    """对单节点进行Lasso回归"""
    # model1 = LassoCV(cv = 5, fit_intercept=True)
    # model2 = OrthogonalMatchingPursuitCV(cv = 5, fit_intercept=True)
    model3 = ARDRegression(max_iter=1000,threshold_lambda=1e4,alpha_1=1e-6,alpha_2=1e-6,lambda_1=1e-6,lambda_2=1e-6, fit_intercept=True)
    # model1_0 = LassoCV(cv = 5,  fit_intercept=False)
    # model2_0 = OrthogonalMatchingPursuitCV(cv = 5, fit_intercept=False)
    model3_0 = ARDRegression(max_iter=1000,threshold_lambda=1e4,alpha_1=1e-6,alpha_2=1e-6,lambda_1=1e-6,lambda_2=1e-6, fit_intercept=False)
    # model1.fit(x_data.cpu(), x_dot.cpu())
    # model2.fit(x_data.cpu(), x_dot.cpu())
    model3.fit(x_data.cpu(), x_dot.cpu())
    # model1_0.fit(x_data.cpu(), x_dot.cpu())
    # model2_0.fit(x_data.cpu(), x_dot.cpu())
    model3_0.fit(x_data.cpu(), x_dot.cpu())
    # model = [model1, model2, model3, model1_0, model2_0, model3_0]
    # aic = [lasso_AIC(model1, x_data.cpu(), x_dot.cpu().numpy()),
    #        lasso_AIC(model2, x_data.cpu(), x_dot.cpu().numpy()),
    #        lasso_AIC(model3, x_data.cpu(), x_dot.cpu().numpy()),
    #        lasso_AIC(model1_0, x_data.cpu(), x_dot.cpu().numpy()),
    #        lasso_AIC(model2_0, x_data.cpu(), x_dot.cpu().numpy()),
    #        lasso_AIC(model3_0, x_data.cpu(), x_dot.cpu().numpy())
    #        ]
    model = [model3, model3_0]
    aic = [lasso_AIC(model3, x_data.cpu(), x_dot.cpu().numpy()),
            lasso_AIC(model3_0, x_data.cpu(), x_dot.cpu().numpy())]
    model_index = np.argmin(aic)
    coef = model[model_index].coef_
    coef[np.abs(coef) < 0.001] = 0
    return coef, model[model_index].intercept_


def summarize_basis_functions(coefs_list):
    """汇总所有节点的可能基函数"""
    coefs_matrix = np.array(coefs_list)
    basis_indicator = (np.abs(coefs_matrix) > 1e-3).any(axis=0).astype(int)
    return basis_indicator


def generate_primary_mask(args, batchs):
    print('Start Lasso mask...')
    device = args.device
    batchs = batchs.to(device)
    fun_names = utils.fun_lib(torch.empty((0,1)), args.poly_p, args.poly_n, device="cpu", activate=args.activate, names=True)
    coupled_names = utils.coupled_fun_lib(None, None, args.poly_p, args.poly_n, device="cpu", activate=args.activate, names=True)
    f_num = len(fun_names)
    c_num = len(coupled_names)

    print(batchs)
    data, edge_index, batch, t = batchs.x, batchs.edge_index, batchs.batch, batchs.t.reshape(-1,args.time_stamp)[0]
    nums = min(args.lasso_node_num, data.shape[0])
    random_index = random.sample(list(np.arange(data.shape[0])), int(nums))

    coefs = []
    intercepts = []
    count = 0
    x_data = []
    x_dot = []
    p_num = []
    for i in random_index:
        x_data0, x_dot0 = compute_node_features(data, i, edge_index, t, args, device) 
        if x_data0.shape[1] < f_num:
            continue                       
        coef0, intercept0= lasso_fit_single_node(x_data0, x_dot0)
        coefs.append(np.hstack((coef0, intercept0)))
        intercepts.append(intercept0)
        p_num.append(np.sum(np.abs(coef0)>0))
        x_dot.append(x_dot0)
        x_data.append(x_data0)

    if args.ode_model == 'SIS':
        p_num = np.array(p_num)
        frequency = Counter(p_num)
        # 2. 获取出现频率最高的两个值
        top2 = frequency.most_common(2)
        # 3. 判断前两个频率的差距
        if len(top2) > 1 and abs(top2[0][1] - top2[1][1]) < 2:
            top2_values = [top2[0][0], top2[1][0]]  # 保留前两个频率值
        else:
            top2_values = [top2[0][0]]  # 只保留最频繁的一个值
        print('top2_values:', top2_values)
        # 4. 筛选出这些值
        selected_indices = [i for i, p in enumerate(p_num) if p in top2_values]
    else:
        vectors_array = np.array(coefs)
        normal_coef = (vectors_array - np.mean(vectors_array, 0))/(np.std(vectors_array, 0) + 1e-5)
        dbscan = DBSCAN(eps=0.1, min_samples=nums//10)  # eps是距离阈值，min_samples是簇内最小样本数
        dbscan.fit(normal_coef)
        # labels = dbscan.labels_
        # positive_labels = labels[labels > 0]
        # label_counts = Counter(positive_labels)
        # core_label = label_counts.most_common(1)[0][0]
        selected_indices = [i for i, p in enumerate(dbscan.labels_) if p >= 0]
        print('core node:', len(selected_indices))
    # 筛选符合条件的数据
    filtered_x_data = [x_data[i] for i in selected_indices]
    filtered_x_dot = [x_dot[i] for i in selected_indices]

    # 仅当有数据时才拼接，避免报错
    if filtered_x_data:
        x_data_aggregated = torch.cat(filtered_x_data, dim=0)
        x_dot_aggregated = torch.cat(filtered_x_dot, dim=0)
    else:
        x_data_aggregated = None
        x_dot_aggregated = None
    coef, intercept= lasso_fit_single_node(x_data_aggregated, x_dot_aggregated)
    # final_index = np.argmin(aics)
    # coefs = coefs[final_index]
    # intercept = intercepts[final_index]
    ###"""generate expressions"""
    if True:
        c_mask = torch.zeros(c_num, 1)
        f_mask = torch.zeros(f_num, 1)
        coef = torch.tensor(coef, dtype=torch.float32, requires_grad=False)
        intercept = torch.tensor(intercept, dtype=torch.float32, requires_grad=False)
        f_mask[0, 0] = intercept
        f_mask[1:,0] = coef[:(f_num-1)]
        c_mask[:,0] = coef[(f_num-1):]
        # ####SIGN Mask
        # expression = utils.functions(args.poly_p, args.poly_n, f_mask, c_mask, activate=args.activate)[0]
        # print('basis:'.format(i), expression)
    return f_mask, c_mask

