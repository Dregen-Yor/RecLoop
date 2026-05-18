accelerate launch --num_processes 1 amazon_text2emb.py \
    --dataset Office_Products \
    --root ../../data/Office_Products \
    --plm_checkpoint /home/aizoo/data/teamshare/models/Qwen3-Embedding-4B
