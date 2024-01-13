ode_model=Kuramoto
for coupled in $(seq 0.1 0.1 1) 
do 
python generate_dataset.py  --coupled $coupled --ode_model $ode_model
done

# ode_model=Fitz
# for coupled in $(seq 0.5 0.5 5) 
# do 
# python generate_dataset.py  --coupled $coupled --ode_model $ode_model
# done