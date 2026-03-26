#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8    # Request 12 cores
#$ -l h_rt=1:0:0  # Request 1 hour runtime
#$ -l h_vmem=7.5G   # Request 11GB RAM per core
#$ -l rocky

module load miniforge/24.7.1 
#module load cuda/12.4.0-gcc-12.2.0 

mamba activate pyg1

node_num=1000
# 定义模型和对应的网络映射
declare -A network_map=( ["FHN"]="small_world" ["HR"]="small_world" ["Rossler"]="small_world" )
declare -A dim_map=( ["FHN"]=2 ["HR"]=3 ["Rossler"]=3 )
# 定义所有模型
models=("FHN" "HR" "Rossler")

# 定义 ob 级别
ob_levels=(0.8 0.85 0.90 0.95 0.99)


# 遍历所有模型
for model in "${models[@]}"; do
    # 获取对应的网络结构
    network=${network_map[$model]}
    init_dim=${dim_map[$model]}
    # 遍历所有 ob 级别
    for ob in "${ob_levels[@]}"; do
        echo "Running simulation with model: $model, network: $network, ob: $ob rate"
        python /data/home/acw802/demo_try/HiSIGN/SIGN_original/SIGN_lasso_hd_missnodes/data/generate_dataset.py --model_name "$model" --network "$network" --ob_node_rate "$ob" --init_dim "$init_dim" --node_num "$node_num"
    done
done

node_num=100000
# 定义模型和对应的网络映射
declare -A network_map=( ["FHN"]="small_world" ["HR"]="small_world" ["Rossler"]="small_world" )
declare -A dim_map=( ["FHN"]=2 ["HR"]=3 ["Rossler"]=3 )
# 定义所有模型
models=("FHN" "HR" "Rossler")

# 定义 ob 级别
ob_levels=(0.8 0.85 0.90 0.95 0.99)


# 遍历所有模型
for model in "${models[@]}"; do
    # 获取对应的网络结构
    network=${network_map[$model]}
    init_dim=${dim_map[$model]}
    # 遍历所有 ob 级别
    for ob in "${ob_levels[@]}"; do
        echo "Running simulation with model: $model, network: $network, ob: $ob rate"
        python /data/home/acw802/demo_try/HiSIGN/SIGN_original/SIGN_lasso_hd_missnodes/data/generate_dataset.py --model_name "$model" --network "$network" --ob_node_rate "$ob" --init_dim "$init_dim" --node_num "$node_num"
    done
done
