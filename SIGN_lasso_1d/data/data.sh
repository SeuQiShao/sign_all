#ode_model=Gene
for ode_model in Kuramoto
do 
sample_num=100
for n in 10 100
do
python generate_dataset.py  --n $n --sample_num $sample_num  --ode_model $ode_model
done
sample_num=10
for n in 1000 10000
do
python generate_dataset.py  --n $n --sample_num $sample_num  --ode_model $ode_model
done
sample_num=1
n=100000
python generate_dataset.py  --n $n --sample_num $sample_num  --ode_model $ode_model
done