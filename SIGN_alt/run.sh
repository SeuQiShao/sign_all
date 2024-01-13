# ode_model=Gene
# num_atoms=1000
# for seed in 3
# do
# python trainer.py    --ode_model $ode_model --seed $seed --num_atoms $num_atoms
# done

# ode_model=Gene
# num_atoms=100000
# for seed in 3
# do
# python trainer.py    --ode_model $ode_model --seed $seed --num_atoms $num_atoms
# done

ode_model=Mutual
num_atoms=1000
for seed in 3
do
python trainer.py    --ode_model $ode_model --seed $seed --num_atoms $num_atoms
done

# ode_model=Mutual
# num_atoms=100000
# for seed in 3
# do
# python trainer.py    --ode_model $ode_model --seed $seed --num_atoms $num_atoms
# done

