# !/bin/bash

# printf "dprnn-Vox2-training \n"
# python ../main.py \
# --save_path exps/dprnn \
# --data_list data_list.csv \
# --audio_path /home/user/data08/VoxCeleb2/wav \
# --visual_path /home/user/data08/VoxMix/lip_open_source \
# --backbone dprnn \
# --n_cpu 12 \
# --length 4 \
# --batch_size 40 \
# --max_epoch 150

# printf "muse-Vox2-training \n"
# python ../main.py \
# --save_path exps/muse \
# --data_list data_list.csv \
# --audio_path /home/user/data08/VoxCeleb2/wav \
# --visual_path /home/user/data08/VoxMix/lip_open_source \
# --backbone muse \
# --n_cpu 12 \
# --length 4 \
# --batch_size 10 \
# --max_epoch 150 \
# --lr 0.0001500

# printf "avsep-Vox2-training \n"
# python ../main.py \
# --save_path exps/avsep \
# --data_list data_list.csv \
# --audio_path /home/user/data08/VoxCeleb2/wav \
# --visual_path /home/user/data08/VoxMix/lip_open_source \
# --backbone avsep \
# --n_cpu 12 \
# --length 4 \
# --batch_size 8 \
# --max_epoch 150 \
# --lr 0.0001500 \

printf "seanet-Vox2-training \n"
python ../main.py \
--save_path exps/seanet \
--data_list data_list.csv \
--audio_path /home/user/data08/VoxCeleb2/wav \
--visual_path /home/user/data08/VoxMix/lip_open_source \
--backbone seanet \
--n_cpu 12 \
--length 4 \
--batch_size 10 \
--max_epoch 150 \
--lr 0.00100