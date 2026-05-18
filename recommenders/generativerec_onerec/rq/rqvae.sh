export CUDA_VISIBLE_DEVICES=1
DATASET_NAME=${DATASET_NAME:-"Toys_and_Games"}
python rqvae.py \
      --data_path ../data/$DATASET_NAME/$DATASET_NAME.emb-sasrec-td.npy \
      --ckpt_dir ./output/$DATASET_NAME/sasrec \
      --lr 1e-3 \
      --epochs 10000 \
      --batch_size 20480 \
      --layers 512 256 128 64

