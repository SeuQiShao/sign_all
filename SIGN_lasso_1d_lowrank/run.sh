#!/bin/bash

# 定义模型和对应的网络映射
declare -A network_map=( ["SIS"]="power_law" ["Kuramoto"]="small_world" ["Gene"]="small_world" )

# 定义所有模型
models=("SIS" "Kuramoto" "Gene")

# 定义 SNR 级别
snr_levels=(20 30 40 50 60 70)

# 遍历所有模型
for model in "${models[@]}"; do
    # 获取对应的网络结构
    network=${network_map[$model]}
    
    # 遍历所有 SNR 级别
    for snr in "${snr_levels[@]}"; do
        echo "Running simulation with model: $model, network: $network, SNR: $snr dB"
        python generate_dataset.py --ode_model "$model" --network "$network" --obnoise "$snr"
    done
done
