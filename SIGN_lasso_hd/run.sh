decoder=DGSI

network=small_world
ode_model=HeatDiffusion
for num_atoms in 10 100 1000 10000 100000
do
python trainer.py   --network $network --ode_model $ode_model --decoder $decoder --num_atoms $num_atoms 
done



network=power_law
ode_model=Kuramoto
for num_atoms in 10 100 1000 10000 100000
do
python trainer.py   --network $network --ode_model $ode_model --decoder $decoder --num_atoms $num_atoms 
done

