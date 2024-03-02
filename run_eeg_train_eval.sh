dataset_choice='/home/jding/Music/BCI_Project/data/'
n_epoch=500
for i in _
do
    python3 main.py --data_path ${dataset_choice} \
    --n_epoch ${n_epoch}
done