decoder=DGSI
network=power_law
ode_model=Kuramoto
num_atoms=100000
for seed in 5 6 7 8 9
do
for noise in $(seq 0.01 0.01 0.3)
do
python trainer.py   --network $network --ode_model $ode_model --seed $seed --num_atoms $num_atoms --noise $noise
done
done
