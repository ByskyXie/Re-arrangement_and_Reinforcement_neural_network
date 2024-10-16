export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u run4RE.py \
  --is_training 1 \
  --root_path ./dataset/R&E/ \
  --data_path TODOOOOOOOOOOOOOOOOOOOOOOOOOOOO \
  --model_id Abilene \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 8 \
  --pred_len 1 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --itr 1
