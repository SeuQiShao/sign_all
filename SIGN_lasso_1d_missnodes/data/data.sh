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
declare -A network_map=( ["SIS"]="power_law" ["Kuramoto"]="small_world" ["Gene"]="small_world" )
declare -A init_map=( ["SIS"]=0.5 ["Kuramoto"]=10 ["Gene"]=1 )
# 定义所有模型
models=("SIS" "Kuramoto" "Gene")

# 定义 ob 级别
ob_levels=(0.8 0.85 0.90 0.95 0.99)


# 遍历所有模型
for model in "${models[@]}"; do
    # 获取对应的网络结构
    network=${network_map[$model]}
    init_scale=${init_map[$model]}
    # 遍历所有 ob 级别
    for ob in "${ob_levels[@]}"; do
        echo "Running simulation with model: $model, network: $network, ob: $ob rate"
        python /data/home/acw802/demo_try/HiSIGN/SIGN_original/SIGN_lasso_1d_missnodes/data/generate_dataset.py --model_name "$model" --network "$network" --ob_node_rate "$ob" --init_scale "$init_scale" --node_num "$node_num"
    done
done

#!/bin/bash
node_num=100000
# 定义模型和对应的网络映射
declare -A network_map=( ["SIS"]="small_world" ["Kuramoto"]="small_world" ["Gene"]="small_world" )
declare -A init_map=( ["SIS"]=0.5 ["Kuramoto"]=10 ["Gene"]=1 )
# 定义所有模型
models=("SIS" "Kuramoto" "Gene")

# 定义 ob 级别
ob_levels=(0.8 0.85 0.90 0.95 0.99)
# 遍历所有模型
for model in "${models[@]}"; do
    # 获取对应的网络结构
    network=${network_map[$model]}
    init_scale=${init_map[$model]}
    # 遍历所有 ob 级别
    for ob in "${ob_levels[@]}"; do
        echo "Running simulation with model: $model, network: $network, ob: $ob rate"
        python /data/home/acw802/demo_try/HiSIGN/SIGN_original/SIGN_lasso_1d_missnodes/data/generate_dataset.py --model_name "$model" --network "$network" --ob_node_rate "$ob" --init_scale "$init_scale" --node_num "$node_num"
    done
done