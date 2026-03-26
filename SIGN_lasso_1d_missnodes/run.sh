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


seeds=(0 1 2 3 4)
for seed in "${seeds[@]}"; do
    # 定义模型和对应的网络映射
    declare -A network_map=( ["SIS"]="power_law" ["Kuramoto"]="small_world" ["Gene"]="small_world" )
    declare -A ac_map=( ["SIS"]='False' ["Kuramoto"]='False' ["Gene"]='True' )
    # 定义所有模型
    models=("SIS" "Kuramoto" "Gene")

    # 定义 ob 级别
    num_atoms=(800 850 900 950 990)


    # 遍历所有模型
    for model in "${models[@]}"; do
        # 获取对应的网络结构
        network=${network_map[$model]}
        activate=${ac_map[$model]}
        # 遍历所有 ob 级别
        for num_atom in "${num_atoms[@]}"; do
            echo "Running  model: $model, network: $network, nodes: $num_atom"
            python /data/home/acw802/demo_try/HiSIGN/SIGN_original/SIGN_lasso_1d_missnodes/trainer.py --ode_model "$model" --network "$network" --activate "$activate" --num_atoms "$num_atom" --seed "$seed"
        done
    done
done
