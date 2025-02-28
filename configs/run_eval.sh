# !/bin/bash

for i in "dprnn" "muse" "avsep" "seanet"; do
    printf "${i} - Evaluate based on the pretrained models\n"
    python ../main.py \
    --save_path trained_models/eval \
    --init_model trained_models/models/${i}.model \
    --data_list data_list.csv \
    --audio_path /home/user/data08/VoxCeleb2/wav \
    --visual_path /home/user/data08/VoxMix/lip_open_source \
    --backbone ${i} \
    --eval
done