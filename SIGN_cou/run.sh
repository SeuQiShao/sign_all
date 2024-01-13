decoder=DGSI
network=power_law
ode_model=Kuramoto
num_atoms=100000
for seed in 0 1 2 3 4 
do
for coupled in $(seq 0.1 0.1 1)
do
python trainer.py   --network $network --ode_model $ode_model --seed $seed --num_atoms $num_atoms --coupled $coupled
done
done
