decoder=DGSI


for runseed in 1 2 3 4 5
do
network=small_world
ode_model=HeatDiffusion
for num_atoms in 10 100 1000 10000 100000
do
lr=0.001
lr_train=0.005
Add_C=False
epochs=200
python trainer.py   --network $network --ode_model $ode_model --decoder $decoder --seed $runseed --num_atoms $num_atoms --lr $lr --lr_train $lr_train --Add_C $Add_C --epochs $epochs
done
done



for runseed in 1 2 3 4 5
do
network=power_law
ode_model=Kuramoto
for num_atoms in 10 100 1000 10000 100000
do
lr=0.0001
lr_train=0.01
Add_C=True
python trainer.py   --network $network --ode_model $ode_model --decoder $decoder --seed $runseed --num_atoms $num_atoms --lr $lr --lr_train $lr_train --Add_C $Add_C
done
done
